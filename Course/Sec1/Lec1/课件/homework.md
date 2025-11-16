# Linux 和 Git 基础操作作业

## 作业说明

本次作业旨在通过实际操作巩固 Linux 和 Git 的基础知识。请按照题目要求逐步完成各项任务，每个小问都是一个具体的文件操作。

---

## 第一大问：Linux 基础文件操作

### 1.1 创建目录结构

在当前目录下创建以下目录结构：

- `project/`
  - `src/`
  - `docs/`
  - `data/`

### 1.2 创建文件并写入内容

在 `project/src/` 目录下创建以下文件并写入对应内容：

- `main.py`：内容为 `print("Hello, World!")`
- `config.txt`：内容为 `version=1.0`
- `readme.md`：内容为 `# Project Readme`

### 1.3 复制和移动文件

- 将 `project/src/main.py` 复制到 `project/data/` 目录下
- 将 `project/src/config.txt` 移动到 `project/docs/` 目录下

### 1.4 查看文件内容

- 使用 `cat` 命令查看 `project/src/readme.md` 的内容
- 使用 `head` 命令查看 `project/data/main.py` 的前 3 行
- 使用 `ls -l` 命令查看 `project/` 目录的详细列表

### 1.5 创建多行文件

在 `project/docs/` 目录下创建 `notes.txt` 文件，内容包含以下三行：

```
第一行：学习 Linux 基础命令
第二行：掌握文件操作
第三行：练习目录管理
```

---

## 第二大问：Linux 进阶操作

### 2.1 文本搜索和处理

- 在 `project/` 目录下递归搜索包含 "Hello" 的文件
- 使用 `grep` 在 `project/docs/notes.txt` 中查找包含 "Linux" 的行，并显示行号
- 使用 `sed` 命令将 `project/docs/notes.txt` 中所有的 "Linux" 替换为 "Unix"

### 2.2 文件权限管理

- 查看 `project/src/main.py` 的当前权限
- 将 `project/src/main.py` 的权限修改为 `755`（所有者可读写执行，组和其他用户可读执行）
- 将 `project/docs/` 目录及其所有文件的权限递归修改为 `644`

### 2.3 查找文件

- 在 `project/` 目录下查找所有 `.txt` 文件
- 在 `project/` 目录下查找所有 `.py` 文件
- 统计 `project/` 目录下文件的总数（不包括目录）

### 2.4 管道和重定向

- 将 `ls -l project/src/` 的输出保存到 `project/docs/file_list.txt`
- 使用管道将 `cat project/docs/notes.txt` 的输出通过 `grep` 过滤，只显示包含 "命令" 的行，并将结果追加到 `project/docs/filtered.txt`
- 统计 `project/docs/notes.txt` 文件的行数，并将结果保存到 `project/docs/line_count.txt`

### 2.5 文件排序和去重

在 `project/data/` 目录下创建 `numbers.txt` 文件，内容如下（每行一个数字）：

```
5
2
8
2
1
5
9
```

- 使用 `sort` 命令对 `numbers.txt` 进行数字排序，并将结果保存到 `sorted_numbers.txt`
- 使用 `uniq` 命令去除 `sorted_numbers.txt` 中的重复行，并将结果保存到 `unique_numbers.txt`

---

## 第三大问：Git 版本控制操作

### 3.1 初始化 Git 仓库

- 在 `project/` 目录下初始化一个 Git 仓库
- 配置 Git 用户信息（用户名和邮箱，可以使用测试信息）

### 3.2 添加文件到暂存区

- 查看当前 Git 仓库的状态
- 将所有文件添加到暂存区
- 再次查看状态，确认文件已添加到暂存区

### 3.3 提交更改

- 创建第一次提交，提交信息为 "Initial commit: 添加项目基础文件"
- 查看提交历史，确认提交成功

### 3.4 修改文件并提交

- 修改 `project/src/readme.md`，在文件末尾添加一行：`## 更新日志`
- 查看工作区与暂存区的差异
- 将修改添加到暂存区并提交，提交信息为 "Update: 添加更新日志部分"

### 3.5 查看提交历史

- 使用 `git log --oneline` 查看简化的提交历史
- 使用 `git log` 查看详细的提交历史
- 使用 `git show` 查看最后一次提交的详细信息

### 3.6 查看文件差异

- 使用 `git diff` 查看工作区与暂存区的差异（如果有未暂存的修改）
- 使用 `git diff HEAD` 查看工作区与最后一次提交的差异

---

## 期望的最终文件夹结构

完成所有操作后，你的 `project/` 目录应该具有以下结构：

```
project/
├── .git/                          # Git 仓库目录
├── src/
│   ├── main.py                    # print("Hello, World!")
│   └── readme.md                  # 包含更新日志
├── docs/
│   ├── config.txt                 # version=1.0 (从 src 移动过来)
│   ├── notes.txt                  # 包含三行笔记（已替换 Linux 为 Unix）
│   ├── file_list.txt              # src 目录的文件列表
│   ├── filtered.txt               # 过滤后的 notes.txt 内容
│   └── line_count.txt             # notes.txt 的行数
└── data/
    ├── main.py                    # main.py 的副本
    ├── numbers.txt                # 原始数字文件
    ├── sorted_numbers.txt         # 排序后的数字
    └── unique_numbers.txt         # 去重后的数字
```

**注意：**

- `.git/` 目录是隐藏目录，使用 `ls -a` 可以查看
- 所有文件都应该有正确的权限设置
- Git 仓库应该包含至少两次提交记录

---

## 提交要求

1. 完成所有题目后，在 `project/` 目录下运行 `tree` 命令（如果没有安装 tree，可以使用 `find . -type f` 或 `ls -R`）查看目录结构
2. 运行 `git log --oneline --graph` 查看提交历史图
3. 将 `project/` 目录的完整结构截图或文本输出保存为 `project_structure.txt`

---

## 提示

- 如果遇到问题，可以使用 `man` 命令查看命令的帮助文档，例如：`man ls`
- 使用 `git status` 可以随时查看仓库状态
- 注意文件路径的相对路径和绝对路径的区别
- 操作前可以先使用 `pwd` 确认当前所在目录
