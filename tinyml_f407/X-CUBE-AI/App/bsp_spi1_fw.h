/*
 * bsp_spi1_fw.h —— SPI1 总线 (nRF24L01 + W25Q128 分时复用)
 * 引脚(与报告引脚表一致): SCK=PB3, MISO=PB4, MOSI=PB5, APB2@16MHz, 2MHz 时钟
 * 分时纪律: 访问一方外设前必须确保另一方 CS 拉高(各自驱动内保证)
 */
#ifndef BSP_SPI1_FW_H
#define BSP_SPI1_FW_H

#include "main.h"

void    bsp_spi1_init(void);               /* GPIO AF5 + SPI1 主机 2MHz 模式0 */
uint8_t bsp_spi1_rw(uint8_t byte);         /* 单字节全双工收发 */

#endif
