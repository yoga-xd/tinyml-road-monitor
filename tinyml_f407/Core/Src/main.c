/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "pipeline_fw.h"
#include "mpu6050_fw.h"
#include "lcd.h"
#include "bsp_spi1_fw.h"
#include "w25q128_fw.h"
#include "node_role.h"
#include "nrf24_fw.h"
#include <stdio.h>
#include <string.h>
#include <math.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart1;

/* USER CODE BEGIN PV */
static int s_prod_mode = 1;          /* 1=生产模式(采集→推理→报警) 0=验证模式(PC compare_board 比对) */
#if NODE_ROLE == NODE_ROLE_A
static I2C_HandleTypeDef hi2c2_fw;   /* V3 板: GY-521 @ PB10/PB11, 手动初始化(避免 CubeMX 重新生成导致模板 bug 重现) */
#endif
static int  s_lcd_ok = 0;            /* LCD 初始化成功标志（未初始化时跳过屏幕刷新） */
#if NODE_ROLE == NODE_ROLE_A
static float s_z0 = 0.0f;            /* 开机静态 z 轴均值（采集链验收值, 上屏显示） */
#endif
volatile uint32_t s_boot_stage = 0;  /* 启动阶段号: 卡死后经 DFU 读 RAM 定位 (0x20000000 附近, 地址见 .map) */
static volatile int s_paused = 0; /* KEY0 暂停/恢复标志: 1=暂停采集/推理/遥测(心跳照发但带暂停标志, 供 B 板区分暂停与失联) */
#if NODE_ROLE == NODE_ROLE_A
static uint32_t s_page = 0;          /* 0=实时页, >=1=历史页码(KEY1 循环翻页, uint32 防回绕) */
static uint32_t s_erase_until = 0;   /* ERASED 提示保持截止时刻(#6) */
static uint8_t  s_page_dirty = 0;    /* 页面重绘请求(KEY1/KEY2 触发) */
#endif
static int      s_nrf_ok = 0;        /* nRF 初始化结果(失败时系统继续运行, 仅无线上报不可用) */
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
static void fw_print(const char *s)
{
    HAL_UART_Transmit(&huart1, (const uint8_t *)s, (uint16_t)strlen(s), 200U);
}

/* 板载按键 GPIO: PE2/PE3/PE4 低电平有效+内部上拉(A/B 板通用配置)
 * A 板: KEY0=暂停 KEY1=历史翻页 KEY2=清空 | B 板: KEY1=历史翻页 KEY2=清空(KEY0 空闲) */
static void fw_key_init(void)
{
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_GPIOE_CLK_ENABLE();
    g.Pin  = GPIO_PIN_2 | GPIO_PIN_3 | GPIO_PIN_4;   /* KEY2=PE2 KEY1=PE3 KEY0=PE4(低有效, 内部上拉) */
    g.Mode = GPIO_MODE_INPUT;
    g.Pull = GPIO_PULLUP;
    HAL_GPIO_Init(GPIOE, &g);
}

/* 报警输出钩子: 写板载 Flash(传感节点存储) + nRF 发送事件帧(无线上报, 协议 v4 15B 含频谱) */
void alarm_output(const alarm_event_t *ev)
{
    uint8_t rec[16];
    rec[0] = 0xAA;
    rec[1] = ev->cls;
    rec[2] = (uint8_t)(ev->conf * 100.0f + 0.5f);
    rec[3] = 0;
    rec[4] = (uint8_t)(ev->window_id >> 24); rec[5] = (uint8_t)(ev->window_id >> 16);
    rec[6] = (uint8_t)(ev->window_id >> 8);  rec[7] = (uint8_t)(ev->window_id);
    rec[8]  = (uint8_t)(ev->tick >> 24); rec[9]  = (uint8_t)(ev->tick >> 16);
    rec[10] = (uint8_t)(ev->tick >> 8);  rec[11] = (uint8_t)(ev->tick);
    rec[12] = 0; rec[13] = 0; rec[14] = 0; rec[15] = 0;
    w25_store_alarm_record(rec);
    {
        /* 无线事件帧 15B: [0]=0xAA [1]=类别 [2]=置信度x100 [3..6]=tick [7..14]=8×uint8 频段能量 */
        uint8_t ef[15];
        uint32_t t = ev->tick;
        ef[0] = 0xAA;
        ef[1] = ev->cls;
        ef[2] = rec[2];
        ef[3] = (uint8_t)(t >> 24); ef[4] = (uint8_t)(t >> 16);
        ef[5] = (uint8_t)(t >> 8);  ef[6] = (uint8_t)(t);
        pipeline_band_e8(ef + 7);
        nrf_tx_event(ef, 15);
    }
}

/* ==================== 板载 UI 公共助手 (320x480 竖屏) ==================== */
static const char *ui_cls_name(uint8_t i)
{
    return (i == 1) ? "HOLE" : (i == 2) ? "SPEED BUMP"
         : (i == 3) ? "WORN ROAD" : "FLAT";
}

static uint16_t ui_cls_color(uint8_t i)
{
    return (i == 1) ? RED : (i == 2) ? YELLOW
         : (i == 3) ? MAGENTA : GREEN;
}
#if NODE_ROLE == NODE_ROLE_A
/* UI 脏区哨兵(文件级): ui_frame() 重绘静态框架会清掉动态内容区域,
 * 之后必须调用 ui_invalidate() 强制哨兵失效, 否则内容不会重画(缓存失效 bug 修复) */
static int      ui_s_cur = -1, ui_s_ccur = -1;
static int      ui_s_pct[4] = {-1, -1, -1, -1};
static uint32_t ui_s_l1smp = 0xFFFFFFFFU, ui_s_l1inf = 0xFFFFFFFFU;
static uint32_t ui_s_l3alm = 0xFFFFFFFFU, ui_s_l3fl = 0xFFFFFFFFU;
static uint32_t ui_s_l4ok = 0xFFFFFFFFU, ui_s_l4fail = 0xFFFFFFFFU;
static int      ui_s_l5cls = -1;
static uint32_t ui_s_sec = 0xFFFFFFFFU;
static uint8_t  ui_s_force = 0;

static uint32_t ui_tb_sec = 0xFFFFFFFFU;   /* 顶栏哨兵(时钟+PAUSED) */
static int      ui_tb_show = -1;
static void ui_invalidate(void)
{
    ui_s_force = 1;
    ui_tb_sec = 0xFFFFFFFFU;   /* 顶栏哨兵一并失效: ui_frame 会擦掉顶栏 */
    ui_tb_show = -1;
}

/* 顶栏更新: PAUSED 闪烁(1Hz) + 运行时钟(1Hz) —— 实时页与历史页共用同一位置/行为 */
static void ui_topbar_update(void)
{
    char b[32];
    uint32_t sec = HAL_GetTick() / 1000U;
    int show_p = s_paused && ((int)(sec & 1U) != 0);
    if (sec != ui_tb_sec || show_p != ui_tb_show) {
        ui_tb_sec = sec; ui_tb_show = show_p;
        lcd_fill(132, 4, 204, 32, DARKBLUE);
        if (show_p) lcd_show_string(140, 10, lcddev.width, 16, 16, "PAUSED", YELLOW);
        lcd_fill(206, 4, 315, 32, DARKBLUE);
        snprintf(b, sizeof b, "UP %02lu:%02lu:%02lu",
                 (unsigned long)(sec / 3600U), (unsigned long)((sec / 60U) % 60U), (unsigned long)(sec % 60U));
        lcd_show_string(214, 10, lcddev.width, 16, 16, b, WHITE);
    }
}

/* 静态框架: 顶栏/判定卡/概率卡/底栏(自检+标定源), 启动时调用 */
static void ui_frame(void)
{
    char b[96];
    lcd_fill(0, 0, lcddev.width - 1, lcddev.height - 1, WHITE);
    lcd_fill(0, 0, lcddev.width - 1, 35, DARKBLUE);                        /* 顶栏 */
    lcd_show_string(12, 10, lcddev.width, 16, 16, "A-NODE Monitor", WHITE);
    /* 判定卡边框由 ui_refresh 随状态颜色绘制 */
    lcd_show_string(18, 52, lcddev.width, 12, 12, "CLASSIFICATION", BLACK);
    lcd_draw_rectangle(8, 120, 312, 252, GRAYBLUE);                      /* 概率卡(顶边上移给概率行让位) */
    lcd_show_string(18, 124, lcddev.width, 12, 12, "CLASS PROBABILITY", BLACK);
    {
        static const char *nm[4] = {"FLAT", "HOLE", "BUMP", "WORN"};
        for (int i = 0; i < 4; i++) {
            lcd_show_string(18, 137 + i * 30, lcddev.width, 24, 24, (char *)nm[i], BLACK);
            lcd_fill(76, 140 + i * 30, 266, 158 + i * 30, LGRAY);
        }
    }
    lcd_fill(0, 450, lcddev.width - 1, lcddev.height - 1, DARKBLUE);       /* 底栏(单行): 自检+标定源+NRF */
    {   /* 三段连续排列, 间隔统一为 6px(一个空格宽): MPU | NRF | z0 */
        uint8_t id[3]; w25_get_id(id);
        int zi = (int)s_z0;
        int zf = (int)((s_z0 - (float)zi) * 100.0f + 0.5f);
        snprintf(b, sizeof b, "MPU:%02X W25:%02X%02X%02X", mpu6050_get_who(), id[0], id[1], id[2]);
        lcd_show_string(10, 458, lcddev.width, 12, 12, b, WHITE);      /* 17 字 = 102px: 10..112 */
        snprintf(b, sizeof b, "NRF:%s", s_nrf_ok ? "OK" : "ERR");
        lcd_show_string(118, 458, lcddev.width, 12, 12, b, s_nrf_ok ? WHITE : RED);   /* 6 字 = 36px */
        snprintf(b, sizeof b, "z0:%+d.%02dcal", zi, zf);
        lcd_show_string(160, 458, lcddev.width, 12, 12, b, LGRAY);     /* 12 字 = 72px: 160..232 */
    }
}

/* 动态刷新(0.5s 调用): 脏区策略——每区域带哨兵, 只有值变化才重画 */
static void ui_refresh(const pipeline_stats_t *st, const alarm_event_t *ev)
{
    char b[64];
    if (ui_s_force) {                      /* ui_frame 重绘静态框架后: 强制哨兵失效, 本次全量重绘 */
        ui_s_force = 0;
        ui_s_cur = -1; ui_s_ccur = -1;
        ui_s_pct[0] = ui_s_pct[1] = ui_s_pct[2] = ui_s_pct[3] = -1;
        ui_s_l1smp = 0xFFFFFFFFU; ui_s_l1inf = 0xFFFFFFFFU;
        ui_s_l3alm = 0xFFFFFFFFU; ui_s_l3fl = 0xFFFFFFFFU;
        ui_s_l4ok = 0xFFFFFFFFU; ui_s_l4fail = 0xFFFFFFFFU;
        ui_s_l5cls = -1;
        ui_s_sec = 0xFFFFFFFFU;
    }
    uint8_t cur = 0;
    for (int i = 1; i < 4; i++) if (st->last_probs[i] > st->last_probs[cur]) cur = i;
    int ccur = (int)(st->last_probs[cur] * 100.0f + 0.5f);

    if (cur != ui_s_cur || ccur != ui_s_ccur) {                  /* 判定卡: 类别/置信度变化才重画 */
        ui_s_cur = (int)cur; ui_s_ccur = ccur;
        lcd_draw_rectangle(8, 44, 312, 112, ui_cls_color(cur));  /* 边框随状态变色 */
        lcd_fill(10, 62, 310, 110, WHITE);
        lcd_show_string(18, 66, lcddev.width, 32, 32, (char *)ui_cls_name(cur), ui_cls_color(cur));
        snprintf(b, sizeof b, "%d%%", ccur);                     /* 置信度 32pt 右对齐与状态同行 */
        lcd_show_string(302 - strlen(b) * 16, 66, lcddev.width, 32, 32, b, ui_cls_color(cur));
    }

    for (int i = 0; i < 4; i++) {                          /* 概率条: 逐根对比 */
        int pct = (int)(st->last_probs[i] * 100.0f + 0.5f);
        if (pct != ui_s_pct[i]) {
            int t = 140 + i * 30;
            ui_s_pct[i] = pct;
            lcd_fill(76, t, 266, t + 18, LGRAY);
            if (pct > 0) lcd_fill(76, t, 76 + pct * 19 / 10, t + 18, ui_cls_color(i));
            lcd_fill(268, t, 304, t + 18, WHITE);
            snprintf(b, sizeof b, "%d%%", pct);
            lcd_show_string(302 - strlen(b) * 8, t + 1, lcddev.width, 16, 16, b, BLACK);
        }
    }

    if (st->samples != ui_s_l1smp || st->inferences != ui_s_l1inf) {   /* L1: 每刷必变(窄条重画) */
        ui_s_l1smp = st->samples; ui_s_l1inf = st->inferences;
        lcd_fill(10, 258, 230, 281, WHITE);
        snprintf(b, sizeof b, "SAMPLES:%lu", (unsigned long)st->samples);
        lcd_show_string(10, 258, lcddev.width, 24, 24, b, BLACK);
        lcd_fill(10, 288, 260, 311, WHITE);
        snprintf(b, sizeof b, "INFERENCES:%lu", (unsigned long)st->inferences);
        lcd_show_string(10, 288, lcddev.width, 24, 24, b, BLACK);
    }

    if (st->alarms != ui_s_l3alm) {                            /* L3: 本次开机报警数 */
        ui_s_l3alm = st->alarms;
        lcd_fill(10, 318, 220, 341, WHITE);
        snprintf(b, sizeof b, "ALARMS:%lu", (unsigned long)st->alarms);
        lcd_show_string(10, 318, lcddev.width, 24, 24, b, BLACK);
    }

    {
        uint32_t fl = w25_record_count();                   /* L4: FLASH = 历史累计(开机扫描恢复, 断电不丢) */
        if (fl != ui_s_l3fl) {
            ui_s_l3fl = fl;
            lcd_fill(10, 348, 240, 371, WHITE);
            snprintf(b, sizeof b, "FLASH:%lurec", (unsigned long)fl);
            lcd_show_string(10, 348, lcddev.width, 24, 24, b, BLACK);
        }
    }

    {                                                       /* L5: 心跳收发计数 */
        uint32_t tok = 0, tfail = 0;
        nrf_get_tx_stats(&tok, &tfail);
        if (tok != ui_s_l4ok || tfail != ui_s_l4fail) {
            ui_s_l4ok = tok; ui_s_l4fail = tfail;
            lcd_fill(10, 378, 150, 401, WHITE);
            snprintf(b, sizeof b, "TX:%lu/%lu", (unsigned long)tok, (unsigned long)tfail);
            lcd_show_string(10, 378, lcddev.width, 24, 24, b, BLACK);
        }
    }

    if (ev->cls != ui_s_l5cls) {                               /* L6: 锁存报警 */
        ui_s_l5cls = (int)ev->cls;
        lcd_fill(10, 408, 220, 431, WHITE);
        if (ev->cls != 0) {
            snprintf(b, sizeof b, "ALARM: %s", ui_cls_name(ev->cls));
            lcd_show_string(10, 408, lcddev.width, 24, 24, b, RED);
        } else {
            lcd_show_string(10, 408, lcddev.width, 24, 24, "ALARM: none", BLACK);
        }
    }
    ui_topbar_update();                                   /* 顶栏: PAUSED+时钟(共享函数, 实时/历史页同款) */
}

/* 历史报警页: 每页 10 条(16pt), KEY1 循环翻页, 读 W25 记录逐条解析显示 */
static void ui_history_page(void)
{
    char b[96];
    uint32_t n = w25_record_count();
    uint32_t pages = (n + 9) / 10;
    if (pages == 0) pages = 1;
    if (s_page > pages) s_page = pages;
    lcd_fill(0, 44, lcddev.width - 1, 448, WHITE);
    snprintf(b, sizeof b, "ALARM HISTORY  page %lu/%lu", (unsigned long)s_page, (unsigned long)pages);
    lcd_show_string(10, 96, lcddev.width, 16, 16, b, BLUE);
    if (n == 0) {
        lcd_show_string(10, 160, lcddev.width, 24, 24, "no records", LGRAY);
        return;
    }
    uint32_t first = (uint32_t)(s_page - 1) * 10;
    for (int r = 0; r < 10; r++) {
        uint32_t idx = first + (uint32_t)r;
        if (idx >= n) break;
        uint8_t rec[16];
        w25_read_record(idx, rec);
        uint32_t tms = ((uint32_t)rec[8] << 24) | ((uint32_t)rec[9] << 16) | ((uint32_t)rec[10] << 8) | rec[11];
        snprintf(b, sizeof b, "NO.%lu %s %d%% t=%lus",
                 (unsigned long)(idx + 1), ui_cls_name(rec[1]), (int)rec[2],
                 (unsigned long)(tms / 1000U));
        lcd_show_string(10, 124 + r * 24, lcddev.width, 16, 16, b, ui_cls_color(rec[1]));
    }
}


/* V3 板 I2C2 手动初始化: PB10=SCL / PB11=SDA, 100kHz, GY-521 在 ATK-MODULE 口
 * 手写而非走 CubeMX 重新生成——规避 app_x-cube-ai 模板 bug（4 处修复清单）重现 */
static int fw_i2c2_init(void)
{
    GPIO_InitTypeDef g = {0};
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_I2C2_CLK_ENABLE();
    g.Pin       = GPIO_PIN_10 | GPIO_PIN_11;
    g.Mode      = GPIO_MODE_AF_OD;
    g.Pull      = GPIO_PULLUP;
    g.Speed     = GPIO_SPEED_FREQ_HIGH;
    g.Alternate = GPIO_AF4_I2C2;
    HAL_GPIO_Init(GPIOB, &g);

    hi2c2_fw.Instance             = I2C2;
    hi2c2_fw.Init.ClockSpeed      = 100000;
    hi2c2_fw.Init.DutyCycle       = I2C_DUTYCYCLE_2;
    hi2c2_fw.Init.OwnAddress1     = 0;
    hi2c2_fw.Init.AddressingMode  = I2C_ADDRESSINGMODE_7BIT;
    hi2c2_fw.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c2_fw.Init.OwnAddress2     = 0;
    hi2c2_fw.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c2_fw.Init.NoStretchMode   = I2C_NOSTRETCH_DISABLE;
    return (HAL_I2C_Init(&hi2c2_fw) == HAL_OK) ? 0 : -1;
}
#else
/* ==================== B 板角色: 接收/显示/汇聚存储 (320x480 竖屏) ==================== */
static uint8_t  s_b_cls = 0, s_b_conf = 0;     /* 心跳: 当前判定/置信度 */
static uint16_t s_b_zrms = 0;                  /* 心跳: 活动电平 x100 (m/s^2) */
static uint8_t  s_b_seq = 0;                   /* 心跳: 窗序号 */
static uint32_t s_b_hb = 0, s_b_evt = 0, s_b_pkt = 0;
static uint32_t s_b_last_hb = 0;               /* 最近一次心跳的时刻(tick) */
static uint8_t  s_b_evcls = 0;                 /* 最近事件类别(锁存) */
static uint32_t s_b_evtick = 0;                /* 最近事件时刻(tick) */
static uint8_t  s_b_bands[8] = {0,0,0,0,0,0,0,0};  /* 最近事件帧 8 频段能量(和≈255) */
static uint8_t  s_b_flags = 0;                 /* 心跳帧[6] 状态位: bit0 = A 板暂停 */
static uint32_t s_b_lost = 0;                  /* 累计丢帧数(模 256 差值检测, 仅链路连续存活时计) */
static uint8_t  s_b_last_seq = 0;              /* 上一帧心跳序号(丢帧检测基准) */
static uint8_t  s_b_seq_valid = 0;             /* 序号有效性: 1=已收到过首帧 */

/* 页面/按键状态(历史页 + 清空, 与 A 板同款交互) */
static uint32_t s_b_page = 0;                  /* 0=实时页, >=1=历史页码 */
static uint8_t  s_b_page_dirty = 0;            /* 页面重绘请求(KEY1/KEY2 触发) */
static uint32_t s_b_erase_until = 0;           /* ERASED 提示保持截止时刻 */

/* UI 哨兵(文件级): 页面切换后必须 ui_b_invalidate 失效, 否则内容不重画(A 板同款教训) */
static uint8_t  s_b_force = 0;
static uint32_t s_clk_sec = 0xFFFFFFFFU;
static int      s_link = -1, s_cls = -1, s_conf = -1, s_act = -1, s_evcls = -1;
static uint32_t s_hb = 0xFFFFFFFFU, s_evt = 0xFFFFFFFFU, s_fl = 0xFFFFFFFFU, s_seq = 0xFFFFFFFFU;

static void ui_b_invalidate(void)
{
    s_b_force = 1;
    s_clk_sec = 0xFFFFFFFFU;
    s_link = s_cls = s_conf = s_act = s_evcls = -1;
    s_hb = s_evt = s_fl = s_seq = 0xFFFFFFFFU;
}

/* 顶栏时钟(1Hz): 实时页/历史页共用 */
static void ui_clock_update(void)
{
    char b[32];
    uint32_t sec = HAL_GetTick() / 1000U;
    if (sec != s_clk_sec) {
        s_clk_sec = sec;
        lcd_fill(206, 4, 315, 32, DARKBLUE);
        snprintf(b, sizeof b, "UP %02lu:%02lu:%02lu",
                 (unsigned long)(sec / 3600U), (unsigned long)((sec / 60U) % 60U), (unsigned long)(sec % 60U));
        lcd_show_string(214, 10, lcddev.width, 16, 16, b, WHITE);
    }
}

/* 历史报警页(B 板自身 W25 记录): 每页 10 条, KEY1 循环翻页, 与 A 板同款布局 */
static void ui_history_page_b(void)
{
    char b[96];
    uint32_t n = w25_record_count();
    uint32_t pages = (n + 9) / 10;
    if (pages == 0) pages = 1;
    if (s_b_page > pages) s_b_page = pages;
    lcd_fill(0, 44, lcddev.width - 1, 448, WHITE);
    snprintf(b, sizeof b, "ALARM HISTORY  page %lu/%lu", (unsigned long)s_b_page, (unsigned long)pages);
    lcd_show_string(10, 96, lcddev.width, 16, 16, b, BLUE);
    if (n == 0) {
        lcd_show_string(10, 160, lcddev.width, 24, 24, "no records", BLACK);
        return;
    }
    uint32_t first = (uint32_t)(s_b_page - 1) * 10;
    for (int r = 0; r < 10; r++) {
        uint32_t idx = first + (uint32_t)r;
        if (idx >= n) break;
        uint8_t rec[16];
        w25_read_record(idx, rec);
        uint32_t tms = ((uint32_t)rec[8] << 24) | ((uint32_t)rec[9] << 16) | ((uint32_t)rec[10] << 8) | rec[11];
        snprintf(b, sizeof b, "NO.%lu %s %d%% t=%lus",
                 (unsigned long)(idx + 1), ui_cls_name(rec[1]), (int)rec[2],
                 (unsigned long)(tms / 1000U));
        lcd_show_string(10, 124 + r * 24, lcddev.width, 16, 16, b, ui_cls_color(rec[1]));
    }
}

/* 底栏常态显示(NRF 前置): KEY2 提示结束后调用恢复 */
static void b_bottombar_normal(void)
{
    char b[48];
    uint8_t id[3];
    w25_get_id(id);
    lcd_fill(0, 450, lcddev.width - 1, lcddev.height - 1, DARKBLUE);
    snprintf(b, sizeof b, "W25:%02X%02X%02X CH:76 1Mbps", id[0], id[1], id[2]);
    lcd_show_string(10, 458, lcddev.width, 12, 12, b, WHITE);
    snprintf(b, sizeof b, "NRF:%s", s_nrf_ok ? "RX" : "ERR");
    lcd_show_string(148, 458, lcddev.width, 12, 12, b, s_nrf_ok ? WHITE : RED);  /* 跟在 1Mbps 后(间距同前) */
}

/* 底栏临时提示(KEY2 HOLD/ERASING/ERASED): 占用底栏文字区, 操作结束恢复常态 */
static void b_bottombar_msg(const char *s, uint16_t color)
{
    lcd_fill(0, 450, lcddev.width - 1, lcddev.height - 1, DARKBLUE);
    lcd_show_string(10, 458, lcddev.width, 16, 16, (char *)s, color);
}

static void ui_frame_b(void)
{
    lcd_fill(0, 0, lcddev.width - 1, lcddev.height - 1, WHITE);
    lcd_fill(0, 0, lcddev.width - 1, 35, DARKBLUE);
    lcd_show_string(12, 10, lcddev.width, 16, 16, "B-NODE Monitor", WHITE);
    lcd_draw_rectangle(8, 44, 312, 104, GRAYBLUE);      /* LINK 卡(压缩) */
    lcd_show_string(18, 52, lcddev.width, 12, 12, "LINK STATUS", BLACK);
    lcd_draw_rectangle(8, 112, 312, 196, GRAYBLUE);     /* 判定卡(边框由 ui_refresh_b 随类别重画) */
    lcd_show_string(18, 120, lcddev.width, 12, 12, "CURRENT CLASSIFICATION", BLACK);
    lcd_show_string(18, 176, lcddev.width, 12, 12, "ACTIVITY", BLACK);
    lcd_fill(104, 174, 270, 186, LGRAY);
    lcd_draw_rectangle(8, 204, 312, 320, GRAYBLUE);     /* 事件卡(含频谱条形图) */
    lcd_show_string(18, 212, lcddev.width, 12, 12, "LAST ALARM EVENT", BLACK);
    b_bottombar_normal();                               /* 底栏: NRF 前置 + W25 信息 */
}

static void ui_refresh_b(void)
{
    char b[64];
    uint32_t now = HAL_GetTick();

    if (s_b_force) {                     /* 页面切换后: 哨兵失效, 本次全量重绘 */
        s_b_force = 0;
        s_link = s_cls = s_conf = s_act = s_evcls = -1;
        s_hb = s_evt = s_fl = s_seq = 0xFFFFFFFFU;
    }

    {   /* LINK 卡三态: 心跳 2s 内=LIVE(绿)/PAUSED(黄, A 板暂停标志) 否则 LOST(红); AGE 右对齐同行 */
        int link;
        if (!(s_nrf_ok && s_b_last_hb != 0 && (now - s_b_last_hb) < 2000U))
            link = 0;                                     /* LOST: 2s 无心跳 */
        else
            link = (s_b_flags & 1U) ? 2 : 1;              /* PAUSED / LIVE */
        if (link != s_link) {
            s_link = link;
            lcd_fill(10, 62, 310, 100, WHITE);
            if (link == 2)
                lcd_show_string(18, 66, lcddev.width, 32, 32, "PAUSED", YELLOW);
            else if (link == 1)
                lcd_show_string(18, 66, lcddev.width, 32, 32, "CONNECT", GREEN);
            else
                lcd_show_string(18, 66, lcddev.width, 32, 32, "LOST", RED);
        }
        uint32_t age = s_b_last_hb ? (now - s_b_last_hb) : 0;
        if (s_b_last_hb)
            snprintf(b, sizeof b, "AGE %lu.%lu s",
                     (unsigned long)(age / 1000U), (unsigned long)((age % 1000U) / 100U));
        else
            snprintf(b, sizeof b, "AGE --");
        lcd_fill(206, 72, 302, 92, WHITE);
        lcd_show_string(302 - strlen(b) * 8, 76, lcddev.width, 16, 16, b, link ? BLACK : RED);
    }

    if (s_b_cls != s_cls || s_b_conf != s_conf) {   /* 判定卡: 边框随类别变色 + 类别/置信度同行右对齐 */
        s_cls = s_b_cls; s_conf = s_b_conf;
        lcd_draw_rectangle(8, 112, 312, 196, ui_cls_color(s_b_cls));
        lcd_fill(10, 134, 310, 170, WHITE);
        lcd_show_string(18, 134, lcddev.width, 32, 32, (char *)ui_cls_name(s_b_cls), ui_cls_color(s_b_cls));
        snprintf(b, sizeof b, "%d%%", s_b_conf);
        lcd_show_string(302 - strlen(b) * 16, 134, lcddev.width, 32, 32, b, ui_cls_color(s_b_cls));
    }
    {   /* 活动电平条: zRMS 0..2 m/s^2 满量程, 每心跳更新 */
        int act = (int)(s_b_zrms / 2);
        if (act > 100) act = 100;
        if (act != s_act) {
            s_act = act;
            lcd_fill(104, 174, 270, 186, LGRAY);
            if (act > 0) lcd_fill(104, 174, 104 + act * 166 / 100, 186, BLUE);
            lcd_fill(270, 170, 306, 188, WHITE);
            snprintf(b, sizeof b, "%d%%", act);
            lcd_show_string(302 - strlen(b) * 8, 172, lcddev.width, 16, 16, b, BLACK);
        }
    }

    if (s_b_evcls != s_evcls) {                     /* 事件卡: 类别名(类别变化才重画) */
        s_evcls = s_b_evcls;
        lcd_fill(10, 226, 180, 254, WHITE);
        if (s_b_evcls != 0)
            lcd_show_string(18, 228, lcddev.width, 24, 24, (char *)ui_cls_name(s_b_evcls), RED);
        else
            lcd_show_string(18, 228, lcddev.width, 24, 24, "none", BLACK);
    }
    {   /* 事件距今: 右对齐同行(每刷更新) */
        if (s_b_evtick) {
            uint32_t ago = (now - s_b_evtick) / 1000U;
            snprintf(b, sizeof b, "before %lus", (unsigned long)ago);
        } else {
            snprintf(b, sizeof b, "no event yet");
        }
        lcd_fill(180, 230, 306, 250, WHITE);
        lcd_show_string(302 - strlen(b) * 8, 232, lcddev.width, 16, 16, b, BLACK);
    }

    {   /* 计数 4 行(全称 24pt): 链路质量(收/丢) + 事件(收/存) */
        uint32_t fl = w25_record_count();
        if (s_b_hb != s_hb) {
            s_hb = s_b_hb;
            lcd_fill(10, 328, 306, 352, WHITE);
            snprintf(b, sizeof b, "RECEIVED:%lu", (unsigned long)s_b_hb);
            lcd_show_string(10, 328, lcddev.width, 24, 24, b, BLACK);
        }
        if (s_b_lost != s_seq) {                        /* s_seq 哨兵现复用为 LOST 计数哨兵 */
            s_seq = s_b_lost;
            lcd_fill(10, 356, 306, 380, WHITE);
            snprintf(b, sizeof b, "LOST:%lu", (unsigned long)s_b_lost);
            lcd_show_string(10, 356, lcddev.width, 24, 24, b, BLACK);
        }
        if (s_b_evt != s_evt) {
            s_evt = s_b_evt;
            lcd_fill(10, 384, 306, 408, WHITE);
            snprintf(b, sizeof b, "EVENTS:%lu", (unsigned long)s_b_evt);
            lcd_show_string(10, 384, lcddev.width, 24, 24, b, BLACK);
        }
        if (fl != s_fl) {
            s_fl = fl;
            lcd_fill(10, 412, 306, 436, WHITE);
            snprintf(b, sizeof b, "FLASH:%lurec", (unsigned long)fl);
            lcd_show_string(10, 412, lcddev.width, 24, 24, b, BLACK);
        }
    }
    {   /* 频谱条形图: 事件帧 8 频段能量(特征17-24 去直流口径, 和≈255) — 开方刻度, 基线 318 满量程 60px */
        lcd_fill(10, 254, 310, 318, WHITE);
        for (int i = 0; i < 8; i++) {
            int h = (int)(sqrtf(s_b_bands[i] / 255.0f) * 60.0f + 0.5f);
            if (h > 0)
                lcd_fill(14 + i * 36, 318 - h, 14 + i * 36 + 28, 318, ui_cls_color(s_b_evcls));
        }
    }

    ui_clock_update();                             /* 顶栏时钟(1Hz) */
}
#endif
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
/* USER CODE BEGIN ModeSelect */
  if (s_prod_mode) {
#if NODE_ROLE == NODE_ROLE_A
      MX_USART1_UART_Init();
      fw_key_init();
      lcd_init();
      lcd_scan_dir(U2D_R2L);                     /* 默认扫描方向: 用户实测定版 */
      LCD_BL(1);
      lcd_fill(0, 0, lcddev.width - 1, lcddev.height - 1, WHITE);
      lcd_show_string(5, 5, lcddev.width, 16, 16, "booting...", BLACK);
      if (fw_i2c2_init() != 0) {
          lcd_show_string(5, 52, lcddev.width, 16, 16, "[E1] I2C2 INIT FAIL", RED);
          while (1) { }
      }
      {
          int prc = pipeline_init(&hi2c2_fw);
          if (prc != 0) {
              char line[80];
              snprintf(line, sizeof line, "[E2] init rc=%d  who=%02X", prc, mpu6050_get_who());
              lcd_show_string(5, 52, lcddev.width, 16, 16, line, RED);
              while (1) { }
          }
      }
      {
          float z = mpu6050_static_check_z();
          s_z0 = z;
          mpu6050_apply_gravity_calib(z);
      }
      bsp_spi1_init();
      if (w25_init() != 0) {
          uint8_t id[3]; w25_get_id(id);
          char line[80];
          snprintf(line, sizeof line, "[E3] W25 ID=%02X %02X %02X (exp EF 40 18)", id[0], id[1], id[2]);
          lcd_show_string(5, 52, lcddev.width, 16, 16, line, RED);
          while (1) { }
      }
      {
          int nrc = nrf_init();
          s_nrf_ok = (nrc == 0);
          if (nrc != 0) {
              char line[80];
              uint8_t st, aw;
              nrf_get_diag(&st, &aw);
              snprintf(line, sizeof line, "[E4] NRF ERR rc=%d ST=%02X AW=%02X", nrc, st, aw);
              lcd_show_string(5, 52, lcddev.width, 16, 16, line, RED);
              while (1) { }
          }
      }
      s_boot_stage = 6;
      s_lcd_ok = 1;
      ui_frame();
      {
          pipeline_stats_t st0; alarm_event_t ev0;
          memset(&st0, 0, sizeof st0);
          memset(&ev0, 0, sizeof ev0);
          st0.last_probs[0] = 1.0f;
          ui_refresh(&st0, &ev0);
      }
      fw_print("[A][1] pipeline running\r\n");
#else  /* NODE_ROLE_B */
      MX_USART1_UART_Init();
      lcd_init();
      lcd_scan_dir(U2D_R2L);
      LCD_BL(1);
      lcd_fill(0, 0, lcddev.width - 1, lcddev.height - 1, WHITE);
      lcd_show_string(5, 5, lcddev.width, 16, 16, "booting...", BLACK);
      bsp_spi1_init();
      fw_key_init();                             /* KEY1 历史翻页 / KEY2 清空(B 板自身记录) */
      if (w25_init() != 0) {
          uint8_t id[3]; w25_get_id(id);
          char line[80];
          snprintf(line, sizeof line, "[E3] W25 ID=%02X %02X %02X (exp EF 40 18)", id[0], id[1], id[2]);
          lcd_show_string(5, 52, lcddev.width, 16, 16, line, RED);
          while (1) { }
      }
      {
          int nrc = nrf_rx_init();               /* PRX 接收端 */
          s_nrf_ok = (nrc == 0);
          if (nrc != 0) {
              char line[80];
              uint8_t st, aw;
              nrf_get_diag(&st, &aw);
              snprintf(line, sizeof line, "[E4] NRF RX ERR rc=%d ST=%02X AW=%02X", nrc, st, aw);
              lcd_show_string(5, 52, lcddev.width, 16, 16, line, RED);
              while (1) { }
          }
      }
      s_boot_stage = 6;
      s_lcd_ok = 1;
      ui_frame_b();
      ui_refresh_b();
      fw_print("[B][1] rx node running\r\n");
#endif
  }
#if NODE_ROLE == NODE_ROLE_A
  else {
      MX_X_CUBE_AI_Init();   /* 验证模式: 供 PC 端 compare_board.py 比对 */
  }
#endif
  /* USER CODE END ModeSelect */
  /* USER CODE BEGIN 2 */

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

#if NODE_ROLE == NODE_ROLE_A
    if (s_prod_mode) { if (!s_paused) pipeline_process(); } else MX_X_CUBE_AI_Process();
#endif
/* USER CODE BEGIN 3 */
#if NODE_ROLE == NODE_ROLE_A
    if (s_prod_mode) {
        static uint32_t s_last_prn = 0, s_last_hb = 0;
        static uint8_t  s_hb_seq = 0;
        static int cur_cls = 0, cur_conf = 0;   /* static: 心跳帧取最近判定(修复: 原局部变量每轮清零, 心跳恒发 0/0) */
        {   /* KEY0 暂停/恢复 + KEY1 历史翻页 + KEY2 长按 2s 清空(边沿触发+300ms 去抖, #1 修复长按重复) */
            static uint32_t s_k0 = 0, s_k1 = 0, s_k2t0 = 0, s_k2draw = 0, s_k2c_until = 0;
            static uint8_t  s_k2hold = 0, s_prev0 = 1, s_prev1 = 1, s_prev2 = 1;   /* 上次电平: 1=松开 */
            uint32_t now = HAL_GetTick();
            uint8_t cur0 = (HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_4) == GPIO_PIN_RESET);
            uint8_t cur1 = (HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_3) == GPIO_PIN_RESET);
            uint8_t cur2 = (HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_2) == GPIO_PIN_RESET);
            if (cur0 && !s_prev0 && now - s_k0 > 300U) {    /* KEY0 按下沿 */
                s_k0 = now;
                s_paused = !s_paused;
                if (!s_paused) pipeline_resume_reset();
            }
            s_prev0 = cur0;
            if (cur1 && !s_prev1 && now - s_k1 > 300U) {    /* KEY1 按下沿 */
                s_k1 = now;
                uint32_t n = w25_record_count();
                uint32_t pages = (n + 9) / 10;
                if (pages == 0) pages = 1;
                s_page++;
                if (s_page > pages) s_page = 0;   /* 末页再按回实时页 */
                s_page_dirty = 1;
            }
            s_prev1 = cur1;
            if (cur2 && !s_prev2) {                          /* KEY2 按下沿: 进入按住态 */
                s_k2hold = 1;
                s_k2t0 = now;
                s_k2draw = 0;
            }
            s_prev2 = cur2;
            if (s_k2hold) {
                uint32_t held = now - s_k2t0;
                if (!cur2) {                                 /* 提前松手: 取消 */
                    s_k2hold = 0;
                    lcd_fill(10, 433, 310, 449, WHITE);
                    lcd_show_string(10, 433, lcddev.width, 16, 16, "cancel", LGRAY);
                    s_k2c_until = now + 3000U;                 /* #6: cancel 提示 3s 后自动清除 */
                } else if (held >= 2000U) {                  /* 长按 2s 确认 -> 执行清空 */
                    lcd_fill(10, 433, 310, 449, WHITE);
                    lcd_show_string(10, 433, lcddev.width, 16, 16, "ERASING...", RED);
                    w25_erase_records();
                    pipeline_alarm_clear();
                    s_page_dirty = 1;
                    s_erase_until = now + 1500U;
                    s_k2hold = 0;
                } else if (now - s_k2draw > 200U) {          /* 按住期间每 200ms 刷新倒计时 */
                    char b[64];
                    s_k2draw = now;
                    lcd_fill(10, 433, 310, 449, WHITE);
                    snprintf(b, sizeof b, "HOLD %lus TO ERASE ALL...",
                             (unsigned long)((2000U - held + 999U) / 1000U));
                    lcd_show_string(10, 433, lcddev.width, 16, 16, b, RED);
                }
            {   /* #6: cancel 提示 3s 后自动清除 */
                if (s_k2c_until && HAL_GetTick() >= s_k2c_until) {
                    lcd_fill(10, 433, 310, 449, WHITE);
                }
            }
            }
        }
        if (s_page_dirty) {                                /* 页面切换/清空后的重绘 */
            s_page_dirty = 0;
            if (s_page == 0) {
                pipeline_stats_t stp = pipeline_stats();
                alarm_event_t    evp = pipeline_last_alarm();
                ui_invalidate();   /* 框架重绘后哨兵必须失效 */
                ui_frame();
                ui_refresh(&stp, &evp);
            if (s_erase_until) {                                 /* #6: 页面重绘后 ERASED 仍可见 */
                lcd_fill(10, 433, 310, 449, WHITE);
                lcd_show_string(10, 433, lcddev.width, 16, 16, "ERASED", GREEN);
            }
            } else {
                ui_history_page();
            if (s_erase_until) {                                 /* #6: 页面重绘后 ERASED 仍可见 */
                lcd_fill(10, 433, 310, 449, WHITE);
                lcd_show_string(10, 433, lcddev.width, 16, 16, "ERASED", GREEN);
            }
            }
        }
        if (s_nrf_ok && HAL_GetTick() - s_last_hb >= 500U) {  /* 心跳帧(协议 v3: 16B 含活动电平+暂停标志; 暂停时照发维持链路活性) */
            s_last_hb = HAL_GetTick();
            s_hb_seq++;
            nrf_tx_heartbeat((uint8_t)cur_cls, (uint8_t)cur_conf,
                             (uint16_t)(pipeline_activity() * 100.0f + 0.5f), (uint8_t)s_hb_seq,
                             s_paused ? 0x01U : 0x00U);   /* [6] bit0 = 暂停标志 → B 板 LINK 三态 */
        }
        if (HAL_GetTick() - s_last_prn >= 500U) {   /* 0.5s 刷新: INF/TX 每刷 +1, SMP 每刷 +25(50Hz 实况) */
            s_last_prn = HAL_GetTick();
            pipeline_stats_t st = pipeline_stats();
            alarm_event_t    ev = pipeline_last_alarm();
            cur_cls = 0;
            for (int i = 1; i < 4; i++) if (st.last_probs[i] > st.last_probs[cur_cls]) cur_cls = i;
            cur_conf = (int)(st.last_probs[cur_cls] * 100.0f + 0.5f);
            ui_topbar_update();                                 /* 历史页: 顶栏 PAUSED/时钟与实时页同款(#7) */
            if (s_page == 0) ui_refresh(&st, &ev);              /* 实时页周期刷新(v3.2 重构时遗失的调用) */
            if (s_erase_until) {                                 /* #6: ERASED 反馈保持 1.5s */
                if (HAL_GetTick() < s_erase_until) {
                    lcd_fill(10, 433, 310, 449, WHITE);
                    lcd_show_string(10, 433, lcddev.width, 16, 16, "ERASED", GREEN);
                } else {
                    lcd_fill(10, 433, 310, 449, WHITE);          /* 到期清除绿字(原只清标志不清屏) */
                    s_erase_until = 0;
                }
            }
            if (!s_paused) {                              /* 暂停时串口遥测静默 */
                char b[192];
                int c100 = (int)(ev.conf * 100.0f + 0.5f);
                int p0 = (int)(st.last_probs[0] * 100.0f + 0.5f);
                int p1 = (int)(st.last_probs[1] * 100.0f + 0.5f);
                int p2 = (int)(st.last_probs[2] * 100.0f + 0.5f);
                int p3 = (int)(st.last_probs[3] * 100.0f + 0.5f);
                /* 诊断字段: z0=开机静态z  gk=重力标定因子(应≈1.00, 偏离大=开机姿态异常)
                 * std=最近窗std(特征1, <0.15 判静止)  cand=报警候选(类别 计数/3, -1=无候选) */
                int z0i  = (int)(s_z0 * 100.0f + ((s_z0 >= 0) ? 0.5f : -0.5f));
                int z0a  = (z0i < 0) ? -z0i : z0i;
                int gki  = (int)(mpu6050_get_gk() * 100.0f + 0.5f);
                int stdi = (int)(pipeline_window_std() * 1000.0f + 0.5f);
                snprintf(b, sizeof b,
                         "t=%lus s=%lu inf=%lu alm=%lu | cls=%d conf=%d%% | p=[%d %d %d %d]"
                         " | z0=%c%d.%02d gk=%d.%02d std=%d.%03d cand=%d %d/3%s\r\n",
                         (unsigned long)(HAL_GetTick() / 1000U),
                         (unsigned long)st.samples, (unsigned long)st.inferences,
                         (unsigned long)st.alarms,
                         ev.cls, c100, p0, p1, p2, p3,
                         (z0i < 0) ? '-' : '+', z0a / 100, z0a % 100,
                         gki / 100, gki % 100, stdi / 1000, stdi % 1000,
                         pipeline_cand_cls(), pipeline_cand_cnt(),
                         (ev.cls != 0) ? " [ALARM]" : "");
                fw_print(b);
            }
        }
    }
#else  /* NODE_ROLE_B */
    if (s_prod_mode) {
        static uint32_t s_last_prn = 0;
        uint8_t fbuf[16];
        while (s_nrf_ok && nrf_rx_poll(fbuf)) {          /* 排空 RX FIFO(每帧 16B) */
            s_b_pkt++;
            if (fbuf[0] == 0xC3) {                        /* 心跳帧: 判定/置信度/活动电平/序号/状态位 */
                /* 模 256 丢帧检测: 仅在链路连续存活(距上帧 <2s)时进行 ——
                 * A 板重启(序号归零)与断链恢复(帧间隔 >2s)两种场景自动豁免, 不误计 */
                if (s_b_seq_valid && s_b_last_hb != 0 && (HAL_GetTick() - s_b_last_hb) < 2000U) {
                    uint8_t diff = (uint8_t)(fbuf[1] - s_b_last_seq);   /* uint8 减法天然模 256, 回绕自动正确 */
                    if (diff > 1) s_b_lost += diff - 1;
                }
                s_b_last_seq = fbuf[1];
                s_b_seq_valid = 1;
                s_b_seq  = fbuf[1];
                s_b_cls  = fbuf[2];
                s_b_conf = fbuf[3];
                s_b_zrms = ((uint16_t)fbuf[4] << 8) | fbuf[5];
                s_b_flags = fbuf[6];                  /* bit0 = A 板暂停 */
                s_b_hb++;
                s_b_last_hb = HAL_GetTick();
            } else if (fbuf[0] == 0xAA) {                 /* 事件帧(协议 v4 15B): 类别/置信度/tick/8 频段能量 */
                s_b_evt++;
                {
                    uint8_t rec[16];                      /* W25 记录格式不变(16B, window_id 置 0, tick 从帧[3..6] 还原) */
                    rec[0] = 0xAA; rec[1] = fbuf[1]; rec[2] = fbuf[2]; rec[3] = 0;
                    rec[4] = rec[5] = rec[6] = rec[7] = 0;
                    rec[8] = fbuf[3]; rec[9] = fbuf[4]; rec[10] = fbuf[5]; rec[11] = fbuf[6];
                    rec[12] = rec[13] = rec[14] = rec[15] = 0;
                    w25_store_alarm_record(rec);
                }
                for (int i = 0; i < 8; i++) s_b_bands[i] = fbuf[7 + i];
                s_b_evcls  = fbuf[1];
                s_b_evtick = HAL_GetTick();
            }
        }
        {   /* KEY1 历史翻页 + KEY2 长按 2s 清空(B 板自身 W25 记录), 与 A 板同款交互 */
            static uint32_t s_k1 = 0, s_k2t0 = 0, s_k2draw = 0;
            static uint8_t  s_k2hold = 0, s_prev1 = 1, s_prev2 = 1;
            uint32_t now = HAL_GetTick();
            uint8_t cur1 = (HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_3) == GPIO_PIN_RESET);
            uint8_t cur2 = (HAL_GPIO_ReadPin(GPIOE, GPIO_PIN_2) == GPIO_PIN_RESET);
            if (cur1 && !s_prev1 && now - s_k1 > 300U) {    /* KEY1 按下沿 */
                s_k1 = now;
                uint32_t n = w25_record_count();
                uint32_t pages = (n + 9) / 10;
                if (pages == 0) pages = 1;
                s_b_page++;
                if (s_b_page > pages) s_b_page = 0;   /* 末页再按回实时页 */
                s_b_page_dirty = 1;
            }
            s_prev1 = cur1;
            if (cur2 && !s_prev2) {                     /* KEY2 按下沿: 进入按住态 */
                s_k2hold = 1; s_k2t0 = now; s_k2draw = 0;
            }
            s_prev2 = cur2;
            if (s_k2hold) {
                uint32_t held = now - s_k2t0;
                if (!cur2) {                            /* 提前松手: 取消, 底栏恢复常态 */
                    s_k2hold = 0;
                    b_bottombar_normal();
                } else if (held >= 2000U) {             /* 长按确认 → 清空 B 板记录 */
                    b_bottombar_msg("ERASING...", RED);
                    w25_erase_records();
                    s_b_evt = 0; s_b_evcls = 0; s_b_evtick = 0;   /* 计数/锁存一并清零 */
                    s_b_page_dirty = 1;
                    s_b_erase_until = now + 1500U;
                    s_k2hold = 0;
                } else if (now - s_k2draw > 200U) {     /* 按住倒计时(底栏文字区) */
                    char kb[48];
                    s_k2draw = now;
                    snprintf(kb, sizeof kb, "HOLD %lus TO ERASE ALL...",
                             (unsigned long)((2000U - held + 999U) / 1000U));
                    b_bottombar_msg(kb, RED);
                }
            }
        }
        if (s_b_page_dirty) {                           /* 页面切换/清空后的重绘 */
            s_b_page_dirty = 0;
            ui_b_invalidate();                          /* 框架重绘后哨兵必须失效(A 板同款教训) */
            if (s_b_page == 0) {
                ui_frame_b();
                ui_refresh_b();
            } else {
                ui_history_page_b();
            }
            if (s_b_erase_until) b_bottombar_msg("ERASED", GREEN);
        }
        if (HAL_GetTick() - s_last_prn >= 500U) {
            s_last_prn = HAL_GetTick();
            if (s_lcd_ok && s_b_page == 0) ui_refresh_b();   /* 历史页冻结实时刷新, 时钟照走 */
            if (s_lcd_ok && s_b_page != 0) ui_clock_update();
            if (s_b_erase_until) {                       /* ERASED 反馈保持 1.5s 后恢复底栏 */
                if (HAL_GetTick() < s_b_erase_until) {
                    b_bottombar_msg("ERASED", GREEN);
                } else {
                    b_bottombar_normal();
                    s_b_erase_until = 0;
                }
            }
            {
                char b[96];
                snprintf(b, sizeof b, "hb=%lu evt=%lu pkt=%lu cls=%u conf=%u seq=%u\r\n",
                         (unsigned long)s_b_hb, (unsigned long)s_b_evt, (unsigned long)s_b_pkt,
                         s_b_cls, s_b_conf, s_b_seq);
                fw_print(b);
            }
        }
    }
#endif
    /* USER CODE END 3 */
  }
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_0) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 115200;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /* USER CODE BEGIN MX_GPIO_Init_2 */

  /* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
