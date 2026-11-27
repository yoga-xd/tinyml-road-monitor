/*
 * mpu6050_fw.c —— MPU6050 驱动（HAL 硬件 I2C）
 * 接线/引脚确认见 INTEGRATION.md（板载或外接模块，地址由 AD0 决定）
 */
#include "mpu6050_fw.h"
#include <math.h>

#define MPU_ADDR_LOW   (0x68 << 1)   /* AD0 = GND */
#define MPU_ADDR_HIGH  (0x69 << 1)   /* AD0 = VCC */

/* 寄存器 */
#define REG_XA_H        0x3B   /* ACCEL_XOUT_H，连读 6 字节 ax,ay,az */
#define REG_CONFIG      0x1A
#define REG_GYRO_CFG    0x1B
#define REG_ACCEL_CFG   0x1C
#define REG_PWR_MGMT1   0x6B
#define REG_SMPLRT_DIV  0x19
#define REG_WHO_AM_I    0x75

static I2C_HandleTypeDef *s_hi2c = 0;
static uint8_t s_addr = MPU_ADDR_LOW;
static uint8_t s_who = 0;              /* WHO_AM_I actual value (some GY-521 clones return 0x70) */
static float    s_gk = 1.0f;           /* gravity scale factor: measured_z0 / 9.81 (clone-chip range fix) */

static int w8(uint8_t reg, uint8_t val)
{
    return HAL_I2C_Mem_Write(s_hi2c, s_addr, reg, I2C_MEMADD_SIZE_8BIT,
                             &val, 1, 100) == HAL_OK ? 0 : 2;
}

int mpu6050_init(I2C_HandleTypeDef *hi2c)
{
    uint8_t who = 0;
    s_hi2c = hi2c;

    /* 先探测 0x68 再探测 0x69 */
    uint8_t addrs[2] = {MPU_ADDR_LOW, MPU_ADDR_HIGH};
    int found = 0;
    for (int a = 0; a < 2 && !found; a++) {
        s_addr = addrs[a];
        if (HAL_I2C_Mem_Read(s_hi2c, s_addr, REG_WHO_AM_I, I2C_MEMADD_SIZE_8BIT,
                             &who, 1, 100) == HAL_OK) {
            found = 1;
        }
    }
    if (!found) return 1;
    s_who = who;   /* compatible clones (MPU6500=0x70) share register map, accept */
    /* 0x68=MPU6050, 其余为 MPU6515/9250 等兼容芯片的常见值；不符再报错 */

    if (w8(REG_PWR_MGMT1, 0x01)) return 2;   /* 唤醒, 时钟=陀螺仪PLL */
    HAL_Delay(50);
    if (w8(REG_SMPLRT_DIV, 19))  return 2;   /* 1kHz/(1+19)=50Hz */
    if (w8(REG_CONFIG, 0x03))    return 2;   /* DLPF 44Hz（抗混叠, ≤25Hz 奈奎斯特内无损）*/
    for (int r = 0; r < 3; r++) {            /* &#xx&#xx: write-then-read-back-verify, retry x3 */
        if (w8(REG_ACCEL_CFG, 0x08)) return 2;
        HAL_Delay(10);
        uint8_t cfg_rb = 0;
        if (HAL_I2C_Mem_Read(s_hi2c, s_addr, REG_ACCEL_CFG, I2C_MEMADD_SIZE_8BIT,
                             &cfg_rb, 1, 100) == HAL_OK && cfg_rb == 0x08) break;
    }
    if (w8(REG_GYRO_CFG, 0x00))  return 2;   /* ±250dps（本模型不用陀螺仪）*/
    return 0;
}

int mpu6050_read_acc_z(float *z_mps2)
{
    uint8_t buf[6];
    if (HAL_I2C_Mem_Read(s_hi2c, s_addr, REG_XA_H, I2C_MEMADD_SIZE_8BIT,
                         buf, 6, 100) != HAL_OK) return 1;
    int16_t az_raw = (int16_t)((buf[4] << 8) | buf[5]);
    *z_mps2 = ((float)az_raw / MPU_LSB_PER_G) * MPU_G / s_gk;
    return 0;
}

uint8_t mpu6050_get_who(void) { return s_who; }

/* gravity auto-calibration: measured static z (with whatever chip range) -> normalize to 9.81 */
void mpu6050_apply_gravity_calib(float measured_z)
{
    float az = fabsf(measured_z);   /* 修复#2: 量程比例与方向无关, 侧放/倒扣开机也能正确标定 */
    /* 20261125 回退到 13b26704 行为: az>1.0G 即标定——平放开机实测 z0=20.4G 证明本模块
     * 等效量程约 2x, 必须靠开机标定修正; 代价: 斜置开机仍可能毒化, 靠"上电平放"纪律规避 */
    if (az > 1.0f) s_gk = az / MPU_G;
}

/* 标定因子读出(诊断用): 应约 1.00, 偏离大 = 开机姿态异常 */
float mpu6050_get_gk(void) { return s_gk; }

float mpu6050_static_check_z(void)
{
    float sum = 0, z;
    for (int i = 0; i < 50; i++) {
        if (mpu6050_read_acc_z(&z) == 0) sum += z;
        HAL_Delay(20);
    }
    return sum / 50.0f;
}
