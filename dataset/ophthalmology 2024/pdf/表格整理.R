# 安装必要包（若未安装）
if (!require("writexl")) install.packages("writexl")
if (!require("fs")) install.packages("fs")

library(writexl)
library(fs)

#---------- 用户必须修改的部分 ----------
# 用实际绝对路径替换下面的路径（获取方法见注释）
main_path <- "E://桌面//case report//JAMA_O_case" # ← 必须修改！
main_path = "E:/桌面/图文大模型/ophthalmology 2024"
#---------------------------------------

# ========== 路径验证阶段 ==========
cat("\n====== 阶段1：路径验证 ======\n")
cat("1. 主路径是否存在：", file.exists(main_path), "\n")
cat("2. 主路径内容：\n")
print(dir_ls(main_path, type = "directory"))  # 使用fs包更清晰的显示

# ========== 文件扫描阶段 ==========
cat("E:/桌面/图文大模型/ophthalmology 2024/PDF")
subfolders <- dir_ls(main_path, type = "directory", recurse = 0)

if (length(subfolders) == 0) {
  stop("错误：主文件夹中没有找到子文件夹！请检查：", main_path)
}

result_df <- data.frame()

# ========== 核心处理逻辑 ==========
for (folder in subfolders) {
  cat("\n>> 正在处理子文件夹：", path_file(folder), "\n")
  
  # 显示文件夹信息
  cat(" - 绝对路径：", folder, "\n")
  cat(" - 是否存在：", file.exists(folder), "\n")
  
  # 扫描PDF文件（包含多重验证）
  pdf_files <- list.files(
    path = folder,
    pattern = "\\.pdf$",
    ignore.case = TRUE,
    full.names = FALSE
  )
  
  # 显示扫描结果
  cat(" - 找到PDF文件数量：", length(pdf_files), "\n")
  if (length(pdf_files) > 0) {
    cat(" - 前5个文件名示例：\n")
    print(head(pdf_files, 5))
  }
  
  # 跳过空文件夹
  if (length(pdf_files) == 0) next
  
  # 生成新文件名
  folder_name <- path_file(folder)
  new_names <- paste0(
    folder_name,
    sprintf("%03d", seq_along(pdf_files)),
    ".", tools::file_ext(pdf_files)
  )
  
  # 安全重命名（避免覆盖）
  for (i in seq_along(pdf_files)) {
    old_file <- path(folder, pdf_files[i])
    new_file <- path(folder, new_names[i])
    
    # 验证单个文件
    if (!file.exists(old_file)) {
      warning("文件不存在：", old_file)
      next
    }
    
    # 执行重命名
    file.rename(old_file, new_file)
    cat(sprintf("重命名成功：%s → %s\n", pdf_files[i], new_names[i]))
  }
  
  # 记录结果
  temp_df <- data.frame(
    原文件名 = pdf_files,
    新文件名 = new_names,
    所属文件夹 = folder_name,
    状态 = ifelse(file.exists(path(folder, new_names)), "成功", "失败"),
    stringsAsFactors = FALSE
  )
  
  result_df <- rbind(result_df, temp_df)
}

# ========== 结果输出 ==========
cat("\n====== 阶段3：结果保存 ==========\n")
output_file <- "PDF重命名记录表.xlsx"
write_xlsx(result_df, output_file)
cat("保存结果到：", normalizePath(output_file), "\n")

# 最终验证
cat("\n====== 处理完成 ======\n")
cat("总处理文件数：", nrow(result_df), "\n")
cat("成功率：", mean(result_df$状态 == "成功")*100, "%\n")
