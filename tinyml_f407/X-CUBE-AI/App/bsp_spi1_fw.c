/*
 * bsp_spi1_fw.c —— SPI1 总线初始化与字节收发
 * nRF24L01(SPI 模式0, CSN=PG7) 与 W25Q128(SPI 模式0, CS=PB14) 分时复用
 * 注意: PB3/PB4 复位默认为 JTAG 复用(AF0), 直接配置为 AF5 即可释放(SWD 调试不受影响)
 */
#include "bsp_spi1_fw.h"

SPI_HandleTypeDef hspi1_fw;

void bsp_spi1_init(void)
{
    GPIO_InitTypeDef g = {0};

    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_SPI1_CLK_ENABLE();

    g.Pin       = GPIO_PIN_3 | GPIO_PIN_4 | GPIO_PIN_5;   /* SCK / MISO / MOSI */
    g.Mode      = GPIO_MODE_AF_PP;
    g.Pull      = GPIO_PULLUP;
    g.Speed     = GPIO_SPEED_FREQ_HIGH;
    g.Alternate = GPIO_AF5_SPI1;
    HAL_GPIO_Init(GPIOB, &g);

    /* W25Q128 片选 PB14 预初始化为无效态(高): 防止其 CS 悬空导致 MISO 总线竞争 */
    {
        GPIO_InitTypeDef cs = {0};
        cs.Pin = GPIO_PIN_14;
        cs.Mode = GPIO_MODE_OUTPUT_PP;
        cs.Pull = GPIO_PULLUP;
        cs.Speed = GPIO_SPEED_FREQ_HIGH;
        HAL_GPIO_Init(GPIOB, &cs);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET);
    }

    /* nRF24L01 CE(PG6)/CSN(PG7) 预初始化为无效态: CSN=高(不选中) CE=低(不发送)
     * 修复: 此前 PG6/PG7 悬空, W25 开机扫描的大量 SPI 时钟可能误选中 nRF,
     * 垃圾字节凑成完整发送序列 -> STATUS 残留 MAX_RT(0x1E) -> init 回读失败 */
    __HAL_RCC_GPIOG_CLK_ENABLE();
    {
        GPIO_InitTypeDef ng = {0};
        ng.Pin   = GPIO_PIN_6 | GPIO_PIN_7;
        ng.Mode  = GPIO_MODE_OUTPUT_PP;
        ng.Pull  = GPIO_PULLUP;
        ng.Speed = GPIO_SPEED_FREQ_HIGH;
        HAL_GPIO_Init(GPIOG, &ng);
        HAL_GPIO_WritePin(GPIOG, GPIO_PIN_6, GPIO_PIN_RESET);   /* CE  = 低 */
        HAL_GPIO_WritePin(GPIOG, GPIO_PIN_7, GPIO_PIN_SET);     /* CSN = 高 */
    }

    hspi1_fw.Instance               = SPI1;
    hspi1_fw.Init.Mode              = SPI_MODE_MASTER;
    hspi1_fw.Init.Direction         = SPI_DIRECTION_2LINES;
    hspi1_fw.Init.DataSize          = SPI_DATASIZE_8BIT;
    hspi1_fw.Init.NSS               = SPI_NSS_SOFT;
    hspi1_fw.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_8;   /* 16MHz/8 = 2MHz */
    hspi1_fw.Init.FirstBit          = SPI_FIRSTBIT_MSB;
    hspi1_fw.Init.CLKPhase          = SPI_PHASE_1EDGE;           /* 模式0: nRF 与 W25Q128 通用 */
    hspi1_fw.Init.CLKPolarity       = SPI_POLARITY_LOW;
    hspi1_fw.Init.CRCPolynomial     = 7;
    HAL_SPI_Init(&hspi1_fw);
}

uint8_t bsp_spi1_rw(uint8_t byte)
{
    uint8_t rx = 0;
    HAL_SPI_TransmitReceive(&hspi1_fw, &byte, &rx, 1, 50);
    return rx;
}
