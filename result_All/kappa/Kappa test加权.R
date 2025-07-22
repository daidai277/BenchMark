# 加载必要的包
library(irr)       # Kappa检验
library(ggplot2)   # 数据可视化
library(vcd)       # 一致性矩阵可视化
library(boot)      # Bootstrap置信区间
library(ggExtra)   # 边缘分布图

# 1. 读取CSV文件（请替换为实际文件路径）
data <- read.csv("Qwen2.5-VL-7B-Instruct .csv", header = TRUE)

# 2. 数据预处理
# 提取评分列并转换为有序因子
ratings <- data[, c("WJM", "ZP")]
ratings$WJM <- factor(ratings$WJM, ordered = TRUE)
ratings$ZP <- factor(ratings$ZP, ordered = TRUE)

# 3. 计算加权Kappa统计量
# 使用平方权重（对较大差异给予更高惩罚）
weighted_kappa <- kappa2(ratings, weight = "squared")

# 4. 使用Bootstrap计算置信区间
set.seed(123)  # 确保结果可重现
kappa_fun <- function(data, indices) {
  d <- data[indices, ]
  k_result <- kappa2(d, weight = "squared")
  return(k_result$value)
}
boot_results <- boot(ratings, kappa_fun, R = 2000)
ci <- boot.ci(boot_results, type = "bca")

# 5. 安全地打印结果
cat("\n===== 加权Kappa分析结果 =====\n")
cat("加权Kappa值:", round(weighted_kappa$value, 3), "\n")

# 安全处理标准误和Z值
if(is.numeric(weighted_kappa$statistic)) {
  cat("标准误:", round(weighted_kappa$statistic, 4), "\n")
} else {
  cat("标准误: 无法计算\n")
}

if(is.numeric(weighted_kappa$z)) {
  cat("Z值:", round(weighted_kappa$z, 2), "\n")
} else {
  cat("Z值: 无法计算\n")
}

# 安全处理P值
if(is.numeric(weighted_kappa$p.value)) {
  cat("P值:", round(weighted_kappa$p.value, 4), "\n")
} else {
  cat("P值: 无法计算\n")
}

# 输出Bootstrap置信区间
if(!is.null(ci$bca)) {
  cat("Bootstrap 95%置信区间: [", round(ci$bca[4], 3), ", ", round(ci$bca[5], 3), "]\n\n")
} else {
  cat("Bootstrap置信区间: 无法计算\n\n")
}

# 6. 一致性解释
interpret_kappa <- function(k) {
  if(is.na(k)) return("无法计算")
  if(k < 0) return("一致性比随机猜测还差")
  else if(k <= 0.2) return("极低一致性")
  else if(k <= 0.4) return("一般一致性")
  else if(k <= 0.6) return("中等一致性")
  else if(k <= 0.8) return("高度一致性")
  else return("几乎完全一致")
}
cat("一致性水平:", interpret_kappa(weighted_kappa$value), "\n")

# 7. 可视化分析

## 7.1 一致性矩阵热力图
conf_matrix <- table(ratings$WJM, ratings$ZP)
conf_df <- as.data.frame(conf_matrix)
colnames(conf_df) <- c("WJM", "ZP", "Count")

ggplot(conf_df, aes(x = WJM, y = ZP, fill = Count)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Count), color = "black", size = 4) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "评分一致性热力图",
       subtitle = paste("加权Kappa =", round(weighted_kappa$value, 3)),
       x = "WJM评分", y = "ZP评分") +
  theme_minimal() +
  theme(legend.position = "right")

## 7.2 评分差异分布图
# 转换为数值型（保留原始顺序）
ratings$WJM_num <- as.numeric(ratings$WJM)
ratings$ZP_num <- as.numeric(ratings$ZP)
ratings$Difference <- ratings$WJM_num - ratings$ZP_num

ggplot(ratings, aes(x = Difference)) +
  geom_histogram(binwidth = 1, fill = "dodgerblue", alpha = 0.7) +
  geom_vline(xintercept = mean(ratings$Difference), 
             color = "red", linetype = "dashed", linewidth = 1) +
  labs(title = "评分差异分布",
       subtitle = paste("平均差异 =", round(mean(ratings$Difference), 3)),
       x = "WJM评分 - ZP评分", y = "频数") +
  theme_bw()

## 7.3 增强型散点图
# 添加相关系数（即使没有Z值）
cor_value <- cor(ratings$WJM_num, ratings$ZP_num, use = "complete.obs")

ggplot(ratings, aes(x = WJM_num, y = ZP_num)) +
  geom_jitter(width = 0.15, height = 0.15, alpha = 0.6, color = "darkgreen") +
  geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
  geom_smooth(method = "lm", se = TRUE, color = "blue") +
  annotate("text", x = min(ratings$WJM_num), y = max(ratings$ZP_num),
           label = paste("相关系数 =", round(cor_value, 3)), 
           hjust = 0, vjust = 1, size = 5) +
  labs(title = "评分者一致性散点图",
       subtitle = paste("加权Kappa =", round(weighted_kappa$value, 3)),
       x = "WJM评分", y = "ZP评分") +
  theme_minimal()

## 7.4 Bland-Altman图（带置信区间）
ratings$Avg <- (ratings$WJM_num + ratings$ZP_num)/2
mean_diff <- mean(ratings$Difference, na.rm = TRUE)
sd_diff <- sd(ratings$Difference, na.rm = TRUE)

# 确保有足够数据点绘制BA图
if(nrow(ratings) > 2 && !is.na(sd_diff)) {
  ba_plot <- ggplot(ratings, aes(x = Avg, y = Difference)) +
    geom_point(alpha = 0.7, color = "purple", size = 3) +
    geom_hline(yintercept = mean_diff, color = "blue", linetype = "dashed", linewidth = 1) +
    geom_hline(yintercept = mean_diff + 1.96*sd_diff, color = "red", linewidth = 0.8) +
    geom_hline(yintercept = mean_diff - 1.96*sd_diff, color = "red", linewidth = 0.8) +
    geom_hline(yintercept = 0, color = "darkgreen", linewidth = 0.8) +
    labs(title = "Bland-Altman分析图", 
         subtitle = "展示评分差异与平均水平的关系",
         x = "平均评分", y = "评分差异(WJM - ZP)") +
    theme_bw()
  
  # 添加标注（如果有足够空间）
  if(diff(range(ratings$Avg)) > 0) {
    ba_plot <- ba_plot +
      annotate("text", x = max(ratings$Avg), y = mean_diff, 
               label = paste("平均差 =", round(mean_diff, 1)), 
               vjust = -1,hjust = 1, color = "blue") +
      annotate("text", x = max(ratings$Avg), y = mean_diff + 1.96*sd_diff, 
               label = "+1.96 SD", vjust = -1, hjust = 1,color = "red") +
      annotate("text", x = max(ratings$Avg), y = mean_diff - 1.96*sd_diff, 
               label = "-1.96 SD", vjust = -1, hjust = 1,color = "red")
  }
  
  print(ba_plot)
} else {
  cat("\n无法绘制Bland-Altman图：数据不足\n")
}


# 8. 保存结果
# 保存统计结果
#sink("weighted_kappa_results.txt")
#cat("===== 加权Kappa分析报告 =====\n")
#cat("分析日期:", format(Sys.Date(), "%Y-%m-%d"), "\n\n")
#cat("样本数量:", nrow(ratings), "\n")
#cat("评分范围:", paste(levels(ratings$WJM), collapse = ", "), "\n\n")
#cat("加权Kappa值:", round(weighted_kappa$value, 3), "\n")
#cat("相关系数:", round(cor_value, 3), "\n\n")
#cat("一致百分比:", round(mean(ratings$WJM == ratings$ZP, na.rm = TRUE)*100, 1), "%\n")
#cat("差异分析:\n")
#cat("平均差异 (WJM - ZP):", round(mean(ratings$Difference, na.rm = TRUE), 3), "\n")
#cat("差异标准差:", round(sd(ratings$Difference, na.rm = TRUE), 3), "\n\n")
#cat("一致性水平:", interpret_kappa(weighted_kappa$value), "\n\n")
#sink()

# 保存图表
#ggsave("consistency_heatmap.png", width = 8, height = 6, dpi = 300)
#ggsave("difference_histogram.png", width = 8, height = 6, dpi = 300)
#ggsave("scatter_plot.png", width = 8, height = 6, dpi = 300)

# 只有成功创建时才保存BA图
#if(exists("ba_plot")) {
#  ggsave("bland_altman_plot.png", plot = ba_plot, width = 8, height = 6, dpi = 300)
#}

#ggsave("marginal_distribution.png", width = 8, height = 6, dpi = 300)

# 完成分析
#cat("\n分析完成！结果已保存到工作目录。")
