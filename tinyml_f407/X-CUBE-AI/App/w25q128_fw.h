/*
 * w25q128_fw.h —— 板载 W25Q128 Flash 驱动 (SPI1, CS=PB14)
 * 用途: 传感节点事件存储(条款: 传感节点存储)
 * 记录格式 16B: [0]=0xAA [1]=cls [2]=conf*100 [3]=0 [4..7]=win_id [8..11]=tick_ms [12..15]=预留
 * 存储布局: 记录区 0x001000 起, 顺序追加; 写入新扇区前自动预擦除(4KB)
 */
#ifndef W25Q128_FW_H
#define W25Q128_FW_H

#include "main.h"

int      w25_init(void);        /* 读 JEDEC ID(期望 EF 40 18), 0=成功 */
void     w25_get_id(uint8_t id[3]);           /* 返回 3 字节 JEDEC ID */
uint32_t w25_record_count(void);              /* 本次上电已存记录数 */
void     w25_store_alarm_record(const uint8_t rec[16]);  /* 追加一条 16B 记录(含预擦除) */

#endif

/* ---- 历史记录管理(KEY1/KEY2) ---- */
void w25_read_record(uint32_t idx, uint8_t out[16]);   /* 读单条记录(idx < 记录数) */
void w25_erase_records(void);                          /* 清空全部记录(擦除已用扇区, 阻塞 <1s) */
