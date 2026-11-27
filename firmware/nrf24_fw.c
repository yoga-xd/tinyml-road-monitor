/*
 * nrf24_fw.c —— nRF24L01+ PTX 发送端驱动
 * 说明: 无接收端(B 板)时, 每次发送后 STATUS 会置 MAX_RT——计 fail 属预期;
 *       B 板到手后, TX_DS 计数即真实送达数(丢包率-距离表的数据源)
 */
#include "nrf24_fw.h"
#include "bsp_spi1_fw.h"

#define NRF_CE_PORT   GPIOG
#define NRF_CE_PIN    GPIO_PIN_6
#define NRF_CSN_PORT  GPIOG
#define NRF_CSN_PIN   GPIO_PIN_7

/* 寄存器命令 */
#define NRF_CMD_R_REG      0x00
#define NRF_CMD_W_REG      0x20
#define NRF_CMD_R_RX_PL    0x61
#define NRF_CMD_W_TX_PAYLOAD 0xA0
#define NRF_CMD_FLUSH_TX   0xE1
#define NRF_CMD_FLUSH_RX   0xE2
#define NRF_CMD_R_RX_PL_WID 0x60   /* DPL: 读接收载荷宽度 */
#define NRF_CMD_ACTIVATE   0x50    /* 老 nRF24L01 激活 FEATURE/DYNPD(Si24R1 不需要, 发了无害) */
#define NRF_CMD_NOP        0xFF

/* 寄存器地址 */
#define NRF_REG_CONFIG     0x00
#define NRF_REG_EN_AA      0x01
#define NRF_REG_EN_RXADDR  0x02
#define NRF_REG_SETUP_AW   0x03
#define NRF_REG_SETUP_RETR 0x04
#define NRF_REG_RF_CH      0x05
#define NRF_REG_RF_SETUP   0x06
#define NRF_REG_STATUS     0x07
#define NRF_REG_RX_ADDR_P0 0x0A
#define NRF_REG_TX_ADDR    0x10
#define NRF_REG_RX_PW_P0   0x11
#define NRF_REG_FIFO       0x17
#define NRF_REG_DYNPD      0x1C
#define NRF_REG_FEATURE    0x1D

/* DPL 自检: 两端写 FEATURE.EN_DPL(0x04)+DYNPD.DPL_P0(0x01) 并回读校验;
 * 回读失败先发 ACTIVATE+0x73 再试(老 nRF24L01 需要; Si24R1 无需但无害);
 * 仍失败 → s_dpl=0, 落静态等宽 15B 兜底(心跳补零) —— 任何模块都可用 */
static uint8_t s_dpl = 0;                 /* 1=动态载荷长度 0=静态等宽 15B */
#define NRF_PL_W          15              /* 静态兜底宽度: 事件帧 15B */

static const uint8_t ADDR5[5] = {0x34, 0x43, 0x10, 0x10, 0x01};
static uint32_t s_tx_ok = 0, s_tx_fail = 0;
static uint8_t  s_diag_st = 0, s_diag_aw = 0;   /* 初始化回读诊断值 */

static void nrf_ce(uint8_t high)
{
    HAL_GPIO_WritePin(NRF_CE_PORT, NRF_CE_PIN,
                      high ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static void nrf_csn(uint8_t high)  /* 1=拉高(释放) 0=拉低(选中) */
{
    HAL_GPIO_WritePin(NRF_CSN_PORT, NRF_CSN_PIN,
                      high ? GPIO_PIN_SET : GPIO_PIN_RESET);
}

static uint8_t nrf_rw(uint8_t b) { return bsp_spi1_rw(b); }

static uint8_t nrf_read_reg(uint8_t reg)
{
    uint8_t v;
    nrf_csn(0);
    nrf_rw(reg);
    v = nrf_rw(NRF_CMD_NOP);
    nrf_csn(1);
    return v;
}

static void nrf_write_reg(uint8_t reg, uint8_t val)
{
    nrf_csn(0);
    nrf_rw(reg | NRF_CMD_W_REG);
    nrf_rw(val);
    nrf_csn(1);
}

static void nrf_write_buf(uint8_t reg, const uint8_t *buf, uint8_t len)
{
    nrf_csn(0);
    nrf_rw(reg | NRF_CMD_W_REG);
    for (uint8_t i = 0; i < len; i++) nrf_rw(buf[i]);
    nrf_csn(1);
}

static void nrf_ce_pulse(void)
{
    nrf_ce(1);
    for (volatile int i = 0; i < 300; i++) { }   /* >10us @16MHz */
    nrf_ce(0);
}

static void nrf_dpl_try(void)
{
    uint8_t r;
    nrf_write_reg(NRF_REG_FEATURE, 0x04);              /* EN_DPL */
    r = nrf_read_reg(NRF_REG_FEATURE);
    if (r != 0x04) {
        nrf_csn(0);
        nrf_rw(NRF_CMD_ACTIVATE);
        nrf_rw(0x73);
        nrf_csn(1);
        nrf_write_reg(NRF_REG_FEATURE, 0x04);
        r = nrf_read_reg(NRF_REG_FEATURE);
    }
    if (r != 0x04) return;
    nrf_write_reg(NRF_REG_DYNPD, 0x01);                /* DPL_P0 */
    s_dpl = (nrf_read_reg(NRF_REG_DYNPD) == 0x01) ? 1U : 0U;
    if (!s_dpl) nrf_write_reg(NRF_REG_FEATURE, 0x00);  /* 半开状态回退清零 */
}

int nrf_init(void)
{
    __HAL_RCC_GPIOG_CLK_ENABLE();
    {
        GPIO_InitTypeDef g = {0};
        g.Pin   = NRF_CE_PIN | NRF_CSN_PIN;
        g.Mode  = GPIO_MODE_OUTPUT_PP;
        g.Pull  = GPIO_PULLUP;
        g.Speed = GPIO_SPEED_FREQ_HIGH;
        HAL_GPIO_Init(GPIOG, &g);
        nrf_ce(0);
        nrf_csn(1);
    }

    HAL_Delay(5);                                  /* 上电复位 1.5ms + 余量 */
    nrf_write_reg(NRF_REG_CONFIG,     0x0A);       /* PRIM_TX + PWR_UP */
    nrf_write_reg(NRF_REG_EN_AA,      0x01);       /* 通道0 自动应答 */
    nrf_write_reg(NRF_REG_EN_RXADDR,  0x01);       /* 通道0 接收(收 ACK 用) */
    nrf_write_reg(NRF_REG_SETUP_AW,   0x03);       /* 5 字节地址 */
    nrf_write_reg(NRF_REG_SETUP_RETR, 0x1A);       /* 重发延时 500us, 最多 10 次 */
    nrf_write_reg(NRF_REG_RF_CH,      76);         /* 2.476GHz */
    nrf_write_reg(NRF_REG_RF_SETUP,   0x07);       /* 1Mbps, 0dBm */
    nrf_write_buf(NRF_REG_RX_ADDR_P0, ADDR5, 5);   /* ACK 回到通道0, 地址须同 TX */
    nrf_write_buf(NRF_REG_TX_ADDR,    ADDR5, 5);
    nrf_csn(0); nrf_rw(NRF_CMD_FLUSH_TX); nrf_csn(1);
    nrf_dpl_try();                                  /* DPL 自检: 成功→7B/15B 变长, 失败→静态 15B 兜底 */

    /* 回读校验: STATUS 上电值 0x0E; SETUP_AW 回读 0x03 */
    nrf_write_reg(NRF_REG_STATUS, 0x70);   /* 清 RX_DR/TX_DS/MAX_RT 残留(防误选中遗留) */
    s_diag_st = nrf_read_reg(NRF_REG_STATUS);
    s_diag_aw = nrf_read_reg(NRF_REG_SETUP_AW);
    return (s_diag_st == 0x0E && s_diag_aw == 0x03) ? 0 : -1;
}

/* 发送前置事务: 处理上一次的发送结果标志, 装载新载荷并打 CE */
static void nrf_load_and_fire(const uint8_t *buf, uint8_t len)
{
    uint8_t st = nrf_read_reg(NRF_REG_STATUS);
    if (st == 0xFF) {   /* 修复#4: 总线无应答(模块缺席)时不计收发, 防计数虚增 */
        nrf_csn(0);
        nrf_rw(NRF_CMD_W_TX_PAYLOAD);
        for (uint8_t i = 0; i < len; i++) nrf_rw(buf[i]);
        nrf_csn(1);
        nrf_ce_pulse();
        return;
    }
    if (st & 0x20) { s_tx_ok++;  nrf_write_reg(NRF_REG_STATUS, 0x20); }   /* TX_DS */
    if (st & 0x10) { s_tx_fail++; nrf_write_reg(NRF_REG_STATUS, 0x10); }  /* MAX_RT */
    if (st & 0x01) {                                   /* TX_FULL 兜底清空 */
        nrf_csn(0); nrf_rw(NRF_CMD_FLUSH_TX); nrf_csn(1);
    }
    nrf_csn(0);
    nrf_rw(NRF_CMD_W_TX_PAYLOAD);
    for (uint8_t i = 0; i < len; i++) nrf_rw(buf[i]);
    nrf_csn(1);
    nrf_ce_pulse();
}

/* 心跳帧(协议 v4): [0]=0xC3 [1]=窗序号 [2]=类别 [3]=置信度x100 [4..5]=活动电平x100 [6]=状态位(bit0=A板暂停)
 * DPL 模式发 7B; 静态兜底模式补零到 15B 与事件帧等宽。暂停时照发心跳维持链路活性 */
void nrf_tx_heartbeat(uint8_t cls, uint8_t conf_x100, uint16_t zrms_x100, uint8_t win_seq, uint8_t flags)
{
    uint8_t hb[16];
    hb[0] = 0xC3;
    hb[1] = win_seq;
    hb[2] = cls;
    hb[3] = conf_x100;
    hb[4] = (uint8_t)(zrms_x100 >> 8);
    hb[5] = (uint8_t)(zrms_x100);
    hb[6] = flags;                      /* bit0 = A 板暂停标志 */
    if (s_dpl) { nrf_load_and_fire(hb, 7); return; }
    for (int i = 7; i < NRF_PL_W; i++) hb[i] = 0;
    nrf_load_and_fire(hb, NRF_PL_W);
}

/* 事件帧(协议 v4): 15B [0]=0xAA [1]=类别 [2]=置信度x100 [3..6]=tick [7..14]=8×uint8 频段能量(和≈255)
 * DPL 模式按 len 原样发; 静态兜底模式 len<15 补零(本协议 len=15 恰好等宽) */
void nrf_tx_event(const uint8_t *frame, uint8_t len)
{
    uint8_t pad[16];
    if (s_dpl) { nrf_load_and_fire(frame, len); return; }
    for (uint8_t i = 0; i < len && i < 16; i++) pad[i] = frame[i];
    for (uint8_t i = len; i < NRF_PL_W; i++) pad[i] = 0;
    nrf_load_and_fire(pad, NRF_PL_W);
}

void nrf_get_diag(uint8_t *st, uint8_t *aw)
{
    *st = s_diag_st;
    *aw = s_diag_aw;
}

void nrf_get_tx_stats(uint32_t *ok, uint32_t *fail)
{
    *ok = s_tx_ok;
    *fail = s_tx_fail;
}



/* ==================== PRX 接收端(B 板角色) ==================== */

/* 接收节点初始化: 信道/地址/速率/自动应答与 A 板 PTX 严格一致, CRC 1 字节一致 */
int nrf_rx_init(void)
{
    __HAL_RCC_GPIOG_CLK_ENABLE();
    {
        GPIO_InitTypeDef g = {0};
        g.Pin = NRF_CE_PIN | NRF_CSN_PIN;
        g.Mode = GPIO_MODE_OUTPUT_PP;
        g.Pull = GPIO_PULLUP;
        g.Speed = GPIO_SPEED_FREQ_HIGH;
        HAL_GPIO_Init(GPIOG, &g);
        nrf_ce(0);
        nrf_csn(1);
    }
    HAL_Delay(5);
    nrf_write_reg(NRF_REG_CONFIG,     0x0B);   /* PRIM_RX + PWR_UP + EN_CRC(1 字节, 对齐 A 板 0x0A) */
    nrf_write_reg(NRF_REG_EN_AA,      0x01);   /* 通道0 自动应答(ACK 回 A 板, 其 TX ok 计数生效) */
    nrf_write_reg(NRF_REG_EN_RXADDR,  0x01);
    nrf_write_reg(NRF_REG_SETUP_AW,   0x03);
    nrf_write_reg(NRF_REG_SETUP_RETR, 0x1A);   /* PRX: ACK 重发延时 */
    nrf_write_reg(NRF_REG_RF_CH,      76);
    nrf_write_reg(NRF_REG_RF_SETUP,   0x07);   /* 1Mbps 0dBm */
    nrf_write_buf(NRF_REG_RX_ADDR_P0, ADDR5, 5);
    nrf_write_reg(NRF_REG_RX_PW_P0,   NRF_PL_W);  /* 静态兜底宽度 15B(DPL 模式下失效) */
    nrf_csn(0); nrf_rw(NRF_CMD_FLUSH_RX); nrf_csn(1);
    nrf_write_reg(NRF_REG_STATUS, 0x70);       /* 清状态残留 */
    nrf_dpl_try();                              /* DPL 自检(与 A 板同一套自检逻辑) */
    s_diag_st = nrf_read_reg(NRF_REG_STATUS);
    s_diag_aw = nrf_read_reg(NRF_REG_SETUP_AW);
    nrf_ce(1);                                  /* PRX: CE 高 = 进入接收态 */
    return (s_diag_st == 0x0E && s_diag_aw == 0x03) ? 0 : -1;
}

/* 轮询接收一帧: 有帧返回 1 并填 buf(16B, 帧尾补零); 无帧返回 0。FIFO 多帧时循环调用即可。
 * DPL 模式先 R_RX_PL_WID 读宽度, >32 为坏包须 Flush_RX 丢弃(手册要求); 静态兜底按 NRF_PL_W 收 */
int nrf_rx_poll(uint8_t buf[16])
{
    uint8_t st = nrf_read_reg(NRF_REG_STATUS);
    if (!(st & 0x40) && ((st >> 1) & 7) == 7) return 0;   /* 无 RX_DR 且 RX_P_NO=111(空) */
    uint8_t w = NRF_PL_W;
    if (s_dpl) {
        nrf_csn(0);
        nrf_rw(NRF_CMD_R_RX_PL_WID);
        w = nrf_rw(NRF_CMD_NOP);
        nrf_csn(1);
        if (w == 0 || w > 32) {                            /* 坏包: 丢弃 */
            nrf_csn(0); nrf_rw(NRF_CMD_FLUSH_RX); nrf_csn(1);
            nrf_write_reg(NRF_REG_STATUS, 0x40);
            return 0;
        }
        if (w > 16) {                                      /* 超缓冲上限(本协议帧型 ≤15B): 丢弃 */
            nrf_csn(0); nrf_rw(NRF_CMD_FLUSH_RX); nrf_csn(1);
            nrf_write_reg(NRF_REG_STATUS, 0x40);
            return 0;
        }
    }
    nrf_csn(0);
    nrf_rw(NRF_CMD_R_RX_PL);
    for (uint8_t i = 0; i < w; i++) buf[i] = nrf_rw(0xFF);
    for (uint8_t i = w; i < 16; i++) buf[i] = 0;           /* 帧尾补零, 调用方按帧类型取字段 */
    nrf_csn(1);
    nrf_write_reg(NRF_REG_STATUS, 0x40);       /* 清 RX_DR */
    return 1;
}
