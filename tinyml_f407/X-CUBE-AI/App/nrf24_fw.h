/*
 * nrf24_fw.h —— nRF24L01+ 发送端驱动 (SPI1, CE=PG6, CSN=PG7, IRQ=PG8 预留)
 * 帧格式(协议 v4, DPL 自检版):
 *   心跳帧 7B: [0]=0xC3 [1]=序号 [2]=cls [3]=conf*100 [4..5]=zRMS*100 [6]=状态位(bit0=A板暂停)
 *   事件帧 15B: [0]=0xAA [1]=cls [2]=conf*100 [3..6]=tick [7..14]=8×uint8 频段能量(和≈255)
 *   init 时 DPL 自检: 写 FEATURE.EN_DPL 回读校验(失败先 ACTIVATE 再试) →
 *   成功=变长发收; 失败=自动降级静态等宽 15B(心跳补零) —— 两种模式频谱功能完整
 * 通道/地址: RF_CH=76, 5 字节地址 34 43 10 10 01 (B 板接收端必须一致)
 */
#ifndef NRF24_FW_H
#define NRF24_FW_H

#include "main.h"

int      nrf_init(void);                /* 寄存器配置+回读校验, 0=成功 */
/* 心跳帧 7B/15B(静态兜底): cls/conf_x100/zrms_x100/窗序号/状态位 */
void     nrf_tx_heartbeat(uint8_t cls, uint8_t conf_x100, uint16_t zrms_x100, uint8_t win_seq, uint8_t flags);
void     nrf_tx_event(const uint8_t *frame, uint8_t len);   /* 发送事件帧(协议 v4, len=15) */
void     nrf_get_diag(uint8_t *st, uint8_t *aw);       /* 初始化回读诊断值 */
void     nrf_get_tx_stats(uint32_t *ok, uint32_t *fail);  /* 自上次查询的收发计数(累计) */

#endif

/* ---- PRX 接收端(B 板角色) ---- */
int  nrf_rx_init(void);                        /* 接收节点初始化(含 DPL 自检, 返回 0=成功) */
int  nrf_rx_poll(uint8_t buf[16]);             /* 轮询收一帧: 1=收到, 0=无; buf 恒 16B, 帧尾补零 */
