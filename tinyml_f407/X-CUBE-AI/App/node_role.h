/*
 * node_role.h —— 双板角色编译开关
 * 同一工程产出两份固件: A 板(传感/推理/上报) 与 B 板(接收/显示/汇聚存储)
 * 硬件完全一致(探索者 F407 + LCD + nRF24L01 + W25Q128), 仅 A 板多一个 GY-521
 */
#ifndef NODE_ROLE_H
#define NODE_ROLE_H

#define NODE_ROLE_A 1     /* A 板: 采集→推理→报警→存储→无线发送 */
#define NODE_ROLE_B 2     /* B 板: 无线接收→显示→事件汇聚存储 */

/* 当前编译目标: 改这一行后重新 Build 即得对应固件 */
#define NODE_ROLE NODE_ROLE_A

#endif
