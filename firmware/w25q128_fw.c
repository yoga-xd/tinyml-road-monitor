/*
 * w25q128_fw.c —— 板载 W25Q128 Flash 驱动
 * W25Q128JV: JEDEC ID = EF 40 18; 页编程 256B; 扇区擦除 4KB
 * 与 nRF24L01 分时复用 SPI1: 本驱动所有操作保证 CS(PB14) 操作前后拉高, 且不与 nRF 交叉调用
 */
#include "w25q128_fw.h"
#include "bsp_spi1_fw.h"

#define W25_CS_PORT   GPIOB
#define W25_CS_PIN    GPIO_PIN_14

#define W25_CMD_WRITE_EN   0x06
#define W25_CMD_READ_SR1   0x05
#define W25_CMD_JEDEC      0x9F
#define W25_CMD_READ       0x03
#define W25_CMD_PAGE_PROG  0x02
#define W25_CMD_SECTOR_ER  0x20

static uint8_t  s_id[3] = {0, 0, 0};
static uint32_t s_count = 0;          /* 本次上电记录数 */
static uint32_t s_erased_sector = 0;              /* 0=从未擦过: 每个扇区首次写入前强制擦除(修复: 旧值 0xFFFFFFFF 导致从不擦除, 记录被旧数据污染) */

#define RECORD_BASE   0x001000UL
#define RECORD_SIZE   16UL

static void w25_cs(uint8_t high)   /* 1=拉高(释放) 0=拉低(选中) */
{
    HAL_GPIO_WritePin(W25_CS_PORT, W25_CS_PIN,
                      high ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static void w25_send_addr(uint32_t addr)
{
    bsp_spi1_rw((uint8_t)(addr >> 16));
    bsp_spi1_rw((uint8_t)(addr >> 8));
    bsp_spi1_rw((uint8_t)(addr));
}

static void w25_write_en(void)
{
    w25_cs(0);
    bsp_spi1_rw(W25_CMD_WRITE_EN);
    w25_cs(1);
}

static void w25_wait_busy(void)
{
    uint32_t t0 = HAL_GetTick();
    w25_cs(0);
    bsp_spi1_rw(W25_CMD_READ_SR1);
    while (bsp_spi1_rw(0xFF) & 0x01) {
        if (HAL_GetTick() - t0 > 500) break;   /* 兜底 500ms */
        HAL_Delay(1);
    }
    w25_cs(1);
}

static void w25_read(uint32_t addr, uint8_t *buf, uint16_t len);  /* forward decl */

/* Boot-time scan of record area to restore historical record count (survives power loss).
 * Records are written sequentially with 0xAA header; first non-0xAA slot is the end.
 * Also restores sector pre-erase bookkeeping. */
static void w25_restore_count(void)
{
    uint8_t buf[RECORD_SIZE];
    s_count = 0;
    for (uint32_t i = 0; i < 60000UL; i++) {
        w25_read(RECORD_BASE + i * RECORD_SIZE, buf, RECORD_SIZE);
        if (buf[0] != 0xAA || buf[12] || buf[13] || buf[14] || buf[15]) break;   /* 头 0xAA 且尾 4 字节=0(写入恒定值), 双重校验挡污染记录 */
        s_count++;
    }
    if (s_count > 0) {
        uint32_t next_sect = (RECORD_BASE + s_count * RECORD_SIZE) >> 12;
        uint32_t prev_sect = (RECORD_BASE + (s_count - 1) * RECORD_SIZE) >> 12;
        s_erased_sector = (next_sect == prev_sect) ? (next_sect + 1) : next_sect;
    } else {
        s_erased_sector = (RECORD_BASE >> 12);   /* 空区/损坏区: 首次写入前擦除该扇区(自愈: 清掉污染数据) */
    }
}

int w25_init(void)
{
    __HAL_RCC_GPIOB_CLK_ENABLE();
    {
        GPIO_InitTypeDef g = {0};
        g.Pin = W25_CS_PIN;
        g.Mode = GPIO_MODE_OUTPUT_PP;
        g.Pull = GPIO_PULLUP;
        g.Speed = GPIO_SPEED_FREQ_HIGH;
        HAL_GPIO_Init(W25_CS_PORT, &g);
        HAL_GPIO_WritePin(W25_CS_PORT, W25_CS_PIN, GPIO_PIN_SET);
    }

    w25_cs(0);
    bsp_spi1_rw(W25_CMD_JEDEC);
    s_id[0] = bsp_spi1_rw(0xFF);            /* 厂商: 期望 0xEF (Winbond) */
    s_id[1] = bsp_spi1_rw(0xFF);            /* 类型: 期望 0x40 */
    s_id[2] = bsp_spi1_rw(0xFF);            /* 容量: 期望 0x18 (16MB) */
    w25_cs(1);
    if (s_id[0] != 0xEF) return -1;
    w25_restore_count();                   /* plan B: FLASH count = historical total (survives reboot) */
    return 0;
}

void w25_get_id(uint8_t id[3])
{
    id[0] = s_id[0]; id[1] = s_id[1]; id[2] = s_id[2];
}

uint32_t w25_record_count(void) { return s_count; }

static void w25_read(uint32_t addr, uint8_t *buf, uint16_t len)
{
    w25_cs(0);
    bsp_spi1_rw(W25_CMD_READ);
    w25_send_addr(addr);
    for (uint16_t i = 0; i < len; i++) buf[i] = bsp_spi1_rw(0xFF);
    w25_cs(1);
}

static void w25_page_program(uint32_t addr, const uint8_t *buf, uint16_t len)
{
    w25_write_en();
    w25_cs(0);
    bsp_spi1_rw(W25_CMD_PAGE_PROG);
    w25_send_addr(addr);
    for (uint16_t i = 0; i < len; i++) bsp_spi1_rw(buf[i]);
    w25_cs(1);
    w25_wait_busy();
}

void w25_store_alarm_record(const uint8_t rec[16])
{
    uint32_t addr  = RECORD_BASE + s_count * RECORD_SIZE;
    uint32_t sect  = addr >> 12;

    if (sect >= s_erased_sector) {          /* 进入新扇区前预擦除 */
        w25_write_en();
        w25_cs(0);
        bsp_spi1_rw(W25_CMD_SECTOR_ER);
        w25_send_addr(sect << 12);
        w25_cs(1);
        w25_wait_busy();
        s_erased_sector = sect + 1;
    }
    w25_page_program(addr, rec, RECORD_SIZE);
    s_count++;
}

/* 读单条历史记录(idx 从 0 起): 历史页显示用 */
void w25_read_record(uint32_t idx, uint8_t out[16])
{
    if (idx >= s_count) { for (int i = 0; i < 16; i++) out[i] = 0; return; }
    w25_read(RECORD_BASE + idx * RECORD_SIZE, out, RECORD_SIZE);
}

/* 清空全部历史记录(KEY2 长按确认后调用): 擦除已用扇区 + 计数/簿记归零
 * 4KB 扇区擦除 45ms 典型/400ms 最大(数据手册), 已用扇区数 = ceil(count*16/4096) */
void w25_erase_records(void)
{
    uint32_t used = 1;
    if (s_count > 0) {
        used = ((RECORD_BASE + s_count * RECORD_SIZE - 1) >> 12) - (RECORD_BASE >> 12) + 1;
    }
    for (uint32_t k = 0; k < used; k++) {
        uint32_t sect = (RECORD_BASE >> 12) + k;
        w25_write_en();
        w25_cs(0);
        bsp_spi1_rw(W25_CMD_SECTOR_ER);
        w25_send_addr(sect << 12);
        w25_cs(1);
        w25_wait_busy();
    }
    s_count = 0;
    s_erased_sector = (RECORD_BASE >> 12) + 1;   /* 刚擦完, 首扇区按已擦处理 */
}
