/*
 * mpu6050_fw.h —— MPU6050 驱动（HAL 硬件 I2C 版）
 * 配置对齐训练数据语义:
 *   50 Hz 采样（SMPLRT_DIV=19, DLPF 44Hz 开启时基带 1kHz/(1+19)=50Hz）
 *   ±8g 量程（覆盖减速带冲击 3-4g）
 *   z 轴朝上（静止读数 +1g ≈ +9.8 m/s²，与训练数据 z 轴 +10.2 同向）
 * 无需零偏校准/重力补偿——逐位置标准化自动吸收直流
 */
#ifndef MPU6050_FW_H
#define MPU6050_FW_H

#include "main.h"      /* 提供 hi2c1 等外设句柄 */

#define MPU_ACCEL_RANGE_G   8.0f
#define MPU_LSB_PER_G       4096.0f          /* ±8g → 4096 LSB/g */
#define MPU_G               9.81f

/* 初始化（传入 CubeMX 生成的 I2C 句柄，如 &hi2c1）
 * 返回 0=成功；1=WHO_AM_I 不符；2=配置写入失败 */
int mpu6050_init(I2C_HandleTypeDef *hi2c);

/* 读取 z 轴加速度（m/s²，含重力）。返回 0=成功 */
int mpu6050_read_acc_z(float *z_mps2);

/* 静态自检: 读取 50 个样本，返回均值（应约 +9.8 m/s²，z 轴朝上时）*/
uint8_t mpu6050_get_who(void);                    /* WHO_AM_I actual value */
void mpu6050_apply_gravity_calib(float measured_z); /* boot-time gravity scale auto-calib */
float mpu6050_static_check_z(void);

float mpu6050_get_gk(void);   /* 诊断: 标定因子应约1.00, 明显偏离=开机姿态异常 */

#endif
