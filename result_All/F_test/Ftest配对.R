# 加载必要的包
library(tidyverse)
library(ggpubr)
library(rstatix)
library(PMCMRplus)

# 1. 读取数据
score_data <- read_csv("score.csv")

# 2. 数据预处理
score_data <- score_data %>%
  mutate(
    Group = as.factor(Group),
    CaseID = as.factor(CaseID)
  )

# 3. 输出基本数据信息
cat("=== 数据概览 ===\n")
cat("数据集行数:", nrow(score_data), "\n")
cat("组别数量:", nlevels(score_data$Group), "\n")
cat("题目数量:", nlevels(score_data$CaseID), "\n")
cat("组别分布:\n")
print(table(score_data$Group))

# 4. 描述性统计分析
cat("\n=== 描述性统计 ===\n")
desc_stats <- score_data %>%
  group_by(Group) %>%
  summarise(
    N = n(),
    Median = median(Sc),
    Q1 = quantile(Sc, 0.25),
    Q3 = quantile(Sc, 0.75),
    Mean = mean(Sc),
    SD = sd(Sc),
    .groups = 'drop'
  )
print(desc_stats)

# 5. Friedman检验
cat("\n=== Friedman检验 ===\n")
friedman_test <- score_data %>% 
  friedman_test(Sc ~ Group | CaseID)
print(friedman_test)

# 6. 可视化1: 数据分布
cat("\n=== 可视化1: 数据分布图 ===\n")
dist_plot <- ggplot(score_data, aes(x = Group, y = Sc, fill = Group)) +
  geom_boxplot(alpha = 0.7, width = 0.6) +
  geom_jitter(width = 0.2, size = 1.8, alpha = 0.5, color = "gray30") +
  geom_hline(yintercept = median(score_data$Sc), linetype = "dashed", color = "red", alpha = 0.5) +
  labs(title = "四组模型评分分布",
       subtitle = "箱线图展示各组评分分布",
       x = "模型组",
       y = "评分") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5))

print(dist_plot)
#ggsave("1_data_distribution.png", dist_plot, width = 8, height = 6, dpi = 300)

# 7. 事后检验（如果Friedman检验显著）
if(friedman_test$p < 0.05) {
  cat("\n=== Friedman检验显著(p < 0.05)，进行事后两两比较 ===\n")
  
  # 使用Nemenyi检验进行两两比较
  posthoc_test <- frdAllPairsNemenyiTest(Sc ~ Group | CaseID, data = score_data)
  pval_matrix <- posthoc_test$p.value
  
  # 提取并打印所有配对比较结果
  comparisons <- combn(levels(score_data$Group), 2, simplify = FALSE)
  
  cat("\n两两比较结果:\n")
  results_df <- data.frame()
  
  for(comp in comparisons) {
    group1 <- comp[1]
    group2 <- comp[2]
    
    # 确保顺序正确（行名在前，列名在后）
    if(group1 %in% rownames(pval_matrix) && group2 %in% colnames(pval_matrix)) {
      pval <- pval_matrix[group1, group2]
    } else if(group2 %in% rownames(pval_matrix) && group1 %in% colnames(pval_matrix)) {
      pval <- pval_matrix[group2, group1]
    } else {
      pval <- NA
    }
    
    if(!is.na(pval)) {
      sig_symbol <- ifelse(pval < 0.001, "***", 
                           ifelse(pval < 0.01, "**",
                                  ifelse(pval < 0.05, "*", "ns")))
      
      cat(sprintf("%s vs %s: p = %.4f %s\n", group1, group2, pval, sig_symbol))
      
      # 添加到结果数据框
      results_df <- rbind(results_df, data.frame(
        group1 = group1,
        group2 = group2,
        p = pval,
        p.signif = sig_symbol
      ))
    }
  }
  
  # 8. 可视化2: 两两比较结果
  cat("\n=== 可视化2: 两两比较结果 ===\n")
  
  # 创建带显著性标记的箱线图
  comparisons_plot <- ggplot(score_data, aes(x = Group, y = Sc, fill = Group)) +
    geom_boxplot(alpha = 0.7, width = 0.6) +
    stat_summary(fun = median, geom = "point", shape = 18, size = 4, color = "black") +
    labs(title = "四组模型评分比较与显著性差异",
         subtitle = paste("Friedman检验: χ²(", friedman_test$df, ") = ", 
                          round(friedman_test$statistic, 2), 
                          ", p = ", format.pval(friedman_test$p, digits = 3)),
         x = "模型组",
         y = "评分",
         caption = "黑色菱形表示中位数\n* p < 0.05, ** p < 0.01, *** p < 0.001") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  # 添加显著性标记
  y_max <- max(score_data$Sc) * 1.1
  y_step <- (max(score_data$Sc) - min(score_data$Sc)) * 0.1
  
  for(i in 1:nrow(results_df)) {
    group1 <- results_df$group1[i]
    group2 <- results_df$group2[i]
    pval <- results_df$p[i]
    sig_symbol <- results_df$p.signif[i]
    
    if(sig_symbol != "ns") {
      comparisons_plot <- comparisons_plot +
        geom_signif(
          comparisons = list(c(group1, group2)),
          annotations = sig_symbol,
          y_position = y_max,
          tip_length = 0.01,
          textsize = 6,
          vjust = 0.5
        )
      
      # 增加y位置，避免重叠
      y_max <- y_max + y_step
    }
  }
  
  print(comparisons_plot)
  ggsave("2_pairwise_comparisons.png", comparisons_plot, width = 9, height = 7, dpi = 300)
  
  # 9. 可视化3: 组间差异热力图
  cat("\n=== 可视化3: 组间差异热力图 ===\n")
  
  # 创建p值矩阵用于热力图
  group_levels <- levels(score_data$Group)
  pval_df <- expand.grid(Group1 = group_levels, Group2 = group_levels) %>%
    mutate(
      p_value = apply(., 1, function(x) {
        if(x[1] == x[2]) return(NA)
        if(x[1] %in% rownames(pval_matrix) && x[2] %in% colnames(pval_matrix)) {
          pval_matrix[x[1], x[2]]
        } else if(x[2] %in% rownames(pval_matrix) && x[1] %in% colnames(pval_matrix)) {
          pval_matrix[x[2], x[1]]
        } else NA
      }),
      log_p = ifelse(is.na(p_value), NA, -log10(p_value)),
      significance = case_when(
        p_value < 0.001 ~ "***",
        p_value < 0.01 ~ "**",
        p_value < 0.05 ~ "*",
        TRUE ~ "ns"
      )
    )
  
  heatmap_plot <- ggplot(pval_df, aes(x = Group1, y = Group2, fill = log_p)) +
    geom_tile(color = "white", size = 1) +
    geom_text(aes(label = significance), color = "black", size = 6) +
    scale_fill_gradient2(
      low = "white", 
      high = "red", 
      na.value = "gray90",
      name = "-log10(p-value)",
      limits = c(0, max(pval_df$log_p, na.rm = TRUE))
    ) +
    labs(title = "组间差异显著性热力图",
         subtitle = "颜色越红表示差异越显著",
         x = "模型组",
         y = "模型组") +
    theme_minimal(base_size = 14) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5),
          panel.grid.major = element_blank())
  
  print(heatmap_plot)
  ggsave("3_significance_heatmap.png", heatmap_plot, width = 8, height = 7, dpi = 300)
  
} else {
  cat("\nFriedman检验不显著(p > 0.05)，无需进行事后检验\n")
  
  # 创建基础箱线图（无显著性标记）
  no_sig_plot <- ggplot(score_data, aes(x = Group, y = Sc, fill = Group)) +
    geom_boxplot(alpha = 0.7) +
    stat_summary(fun = median, geom = "point", shape = 18, size = 4, color = "black") +
    labs(title = "四组模型评分比较",
         subtitle = paste("Friedman检验: χ²(", friedman_test$df, ") = ", 
                          round(friedman_test$statistic, 2), 
                          ", p = ", format.pval(friedman_test$p, digits = 3)),
         x = "模型组",
         y = "评分",
         caption = "黑色菱形表示中位数\n各组间无显著差异") +
    theme_minimal(base_size = 14) +
    theme(legend.position = "none",
          plot.title = element_text(face = "bold", hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5))
  
  print(no_sig_plot)
  ggsave("no_significant_differences.png", no_sig_plot, width = 8, height = 6, dpi = 300)
}

# 10. 可视化4: 评分分布密度图
cat("\n=== 可视化4: 评分分布密度图 ===\n")
density_plot <- ggplot(score_data, aes(x = Sc, fill = Group)) +
  geom_density(alpha = 0.6, color = 1) +             #color：修改线的颜色
  facet_wrap(~ Group, ncol = 3) +
  labs(title = "各组评分分布密度",
       subtitle = "展示每组评分的分布特征",
       x = "评分",
       y = "密度") +
  theme_minimal(base_size = 14) +
  theme(legend.position = "none",
        plot.title = element_text(face = "bold", hjust = 0.5),
        plot.subtitle = element_text(hjust = 0.5),
        strip.text = element_text(size = 12, face = "bold"))


print(density_plot)
#ggsave("4_score_density.png", density_plot, width = 10, height = 8, dpi = 300)


cat("\n分析完成! 所有结果和图表已保存到工作目录。\n")