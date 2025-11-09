# Git 基础语法教程

## 目录

1. [Git 简介](#git-简介)
2. [初始配置](#初始配置)
3. [基本操作](#基本操作)
4. [分支管理](#分支管理)
5. [远程仓库](#远程仓库)
6. [撤销操作](#撤销操作)
7. [查看历史](#查看历史)
8. [标签管理](#标签管理)
9. [常用工作流](#常用工作流)

---

## Git 简介

Git 是一个分布式版本控制系统，用于跟踪文件的变化和协作开发。

### 核心概念

- **仓库（Repository）**：项目的版本控制数据库
- **工作区（Working Directory）**：当前编辑的文件
- **暂存区（Staging Area）**：准备提交的文件
- **提交（Commit）**：保存项目状态的快照
- **分支（Branch）**：独立开发线
- **远程仓库（Remote）**：网络上的仓库副本

---

## 初始配置

### 首次设置

```bash
# 设置用户名
git config --global user.name "Your Name"

# 设置邮箱
git config --global user.email "your.email@example.com"

# 查看配置
git config --list
git config user.name
git config user.email

# 设置默认编辑器
git config --global core.editor "vim"
# 或使用 VS Code
git config --global core.editor "code --wait"
```

### 初始化仓库

```bash
# 在当前目录初始化新仓库
git init

# 克隆远程仓库
git clone https://github.com/user/repo.git
git clone https://github.com/user/repo.git my-project  # 指定目录名
```

---

## 基本操作

### 查看状态

```bash
git status                    # 查看工作区状态
git status -s                 # 简短格式
```

### 添加文件

```bash
git add file.txt              # 添加单个文件
git add *.txt                 # 添加所有 .txt 文件
git add .                     # 添加所有更改的文件
git add -A                    # 添加所有文件（包括删除的）
git add -u                    # 只添加已跟踪的文件
```

### 提交更改

```bash
git commit -m "提交信息"       # 提交暂存区的更改
git commit -am "提交信息"      # 添加并提交（跳过暂存区，仅限已跟踪文件）
git commit --amend            # 修改最后一次提交
git commit --amend -m "新信息" # 修改最后一次提交信息
```

### 查看差异

```bash
git diff                      # 查看工作区与暂存区的差异
git diff --staged             # 查看暂存区与最后一次提交的差异
git diff HEAD                 # 查看工作区与最后一次提交的差异
git diff commit1 commit2      # 查看两次提交的差异
```

### 查看文件

```bash
git ls-files                  # 查看已跟踪的文件
git ls-tree HEAD              # 查看提交中的文件树
```

---

## 分支管理

### 查看分支

```bash
git branch                    # 查看本地分支
git branch -a                 # 查看所有分支（包括远程）
git branch -r                 # 查看远程分支
```

### 创建分支

```bash
git branch new-branch         # 创建新分支
git branch new-branch commit  # 从指定提交创建分支
git checkout -b new-branch    # 创建并切换到新分支
git switch -c new-branch      # 创建并切换到新分支（新语法）
```

### 切换分支

```bash
git checkout branch-name      # 切换到指定分支
git switch branch-name        # 切换到指定分支（新语法）
git checkout -                # 切换到上一个分支
```

### 合并分支

```bash
git merge branch-name         # 合并指定分支到当前分支
git merge --no-ff branch-name # 不使用快进合并
git merge --squash branch-name # 压缩合并（所有提交合并为一个）
```

### 删除分支

```bash
git branch -d branch-name     # 删除已合并的分支
git branch -D branch-name     # 强制删除分支
git push origin --delete branch-name  # 删除远程分支
```

---

## 远程仓库

### 查看远程仓库

```bash
git remote                    # 查看远程仓库名称
git remote -v                 # 查看远程仓库详细信息
git remote show origin        # 查看远程仓库详细信息
```

### 添加远程仓库

```bash
git remote add origin https://github.com/user/repo.git
git remote add upstream https://github.com/original/repo.git  # 添加上游仓库
```

### 修改远程仓库

```bash
git remote set-url origin https://github.com/user/new-repo.git
git remote rename old-name new-name
git remote remove remote-name
```

### 获取和拉取

```bash
git fetch origin              # 获取远程更新（不合并）
git fetch origin branch-name  # 获取指定分支
git pull origin main          # 拉取并合并远程分支
git pull                      # 拉取当前分支的远程更新
git pull --rebase             # 使用 rebase 方式拉取
```

### 推送

```bash
git push origin main          # 推送到远程 main 分支
git push origin branch-name   # 推送到指定分支
git push -u origin main       # 推送并设置上游分支
git push                      # 推送到已设置的上游分支
git push --all                # 推送所有分支
git push --tags               # 推送所有标签
```

---

## 撤销操作

### 撤销工作区更改

```bash
git checkout -- file.txt      # 撤销文件修改（危险！）
git restore file.txt          # 撤销文件修改（新语法）
git restore .                 # 撤销所有文件修改
```

### 撤销暂存区

```bash
git reset HEAD file.txt       # 从暂存区移除文件
git restore --staged file.txt # 从暂存区移除文件（新语法）
git restore --staged .        # 移除所有暂存文件
```

### 撤销提交

```bash
# 撤销最后一次提交，保留更改
git reset --soft HEAD~1      # 撤销提交，保留暂存区
git reset --mixed HEAD~1      # 撤销提交和暂存区，保留工作区
git reset --hard HEAD~1       # 完全撤销（危险！会丢失更改）

# 回退到指定提交
git reset --hard commit-hash
```

### 恢复文件

```bash
git checkout commit-hash -- file.txt  # 从指定提交恢复文件
git restore --source=commit-hash file.txt  # 新语法
```

---

## 查看历史

### 查看提交历史

```bash
git log                       # 查看提交历史
git log --oneline            # 单行显示
git log --graph              # 图形化显示
git log --all --graph        # 显示所有分支的图形
git log -n 5                 # 显示最近5次提交
git log --author="name"      # 按作者筛选
git log --since="2023-01-01" # 按时间筛选
git log --grep="关键词"      # 按提交信息搜索
git log file.txt             # 查看文件的提交历史
```

### 查看提交详情

```bash
git show commit-hash          # 查看指定提交的详细信息
git show HEAD                 # 查看最后一次提交
git show HEAD:file.txt        # 查看提交中的文件内容
```

### 查看文件历史

```bash
git log --follow file.txt    # 跟踪文件重命名历史
git blame file.txt           # 查看文件的每一行是谁修改的
```

---

## 标签管理

### 创建标签

```bash
git tag v1.0.0               # 创建轻量标签
git tag -a v1.0.0 -m "版本1.0.0"  # 创建附注标签
git tag v1.0.0 commit-hash   # 在指定提交创建标签
```

### 查看标签

```bash
git tag                      # 列出所有标签
git tag -l "v1.*"            # 按模式搜索标签
git show v1.0.0              # 查看标签详情
```

### 删除标签

```bash
git tag -d v1.0.0            # 删除本地标签
git push origin --delete v1.0.0  # 删除远程标签
```

### 推送标签

```bash
git push origin v1.0.0       # 推送单个标签
git push origin --tags       # 推送所有标签
```

---

## 常用工作流

### 基本工作流

```bash
# 1. 查看状态
git status

# 2. 添加文件
git add .

# 3. 提交更改
git commit -m "描述性信息"

# 4. 推送到远程
git push origin main
```

### 分支工作流

```bash
# 1. 创建功能分支
git checkout -b feature/new-feature

# 2. 开发和提交
git add .
git commit -m "实现新功能"

# 3. 切换回主分支
git checkout main

# 4. 合并功能分支
git merge feature/new-feature

# 5. 删除功能分支
git branch -d feature/new-feature
```

### 协作工作流

```bash
# 1. 获取远程更新
git fetch origin

# 2. 查看差异
git log HEAD..origin/main

# 3. 合并远程更改
git pull origin main

# 4. 解决冲突（如果有）
# 编辑冲突文件，然后：
git add resolved-file.txt
git commit -m "解决冲突"

# 5. 推送更改
git push origin main
```

---

## 高级操作

### 暂存更改

```bash
git stash                    # 暂存当前更改
git stash save "描述信息"     # 带描述的暂存
git stash list               # 查看暂存列表
git stash pop                # 应用并删除最近的暂存
git stash apply              # 应用暂存但不删除
git stash drop               # 删除最近的暂存
git stash clear              # 清空所有暂存
```

### 变基（Rebase）

```bash
git rebase main              # 将当前分支变基到 main
git rebase -i HEAD~3         # 交互式变基（修改最近3次提交）
git rebase --abort           # 中止变基
git rebase --continue        # 继续变基
```

### 清理

```bash
git clean -n                 # 预览要删除的未跟踪文件
git clean -f                 # 删除未跟踪的文件
git clean -fd                # 删除未跟踪的文件和目录
git clean -x                 # 包括忽略的文件
```

### 子模块

```bash
git submodule add https://github.com/user/repo.git path
git submodule init           # 初始化子模块
git submodule update         # 更新子模块
git submodule update --init --recursive  # 初始化并更新所有子模块
```

---

## 忽略文件

创建 `.gitignore` 文件来忽略不需要版本控制的文件：

```bash
# .gitignore 示例

# 编译产物
*.class
*.o
*.exe

# 依赖目录
node_modules/
venv/
__pycache__/

# IDE 配置
.vscode/
.idea/
*.swp

# 系统文件
.DS_Store
Thumbs.db

# 日志文件
*.log

# 环境变量
.env
.env.local
```

---

## 常用别名

设置 Git 别名可以简化常用命令：

```bash
# 设置别名
git config --global alias.st status
git config --global alias.co checkout
git config --global alias.br branch
git config --global alias.ci commit
git config --global alias.unstage 'reset HEAD --'
git config --global alias.last 'log -1 HEAD'
git config --global alias.visual '!gitk'

# 使用别名
git st    # 等同于 git status
git co main  # 等同于 git checkout main
```

---

## 常见问题解决

### 合并冲突

```bash
# 1. 查看冲突文件
git status

# 2. 编辑冲突文件，解决冲突标记
# <<<<<<< HEAD
# 当前分支的内容
# =======
# 要合并的分支的内容
# >>>>>>> branch-name

# 3. 标记为已解决
git add resolved-file.txt

# 4. 完成合并
git commit
```

### 撤销错误的合并

```bash
git merge --abort            # 中止正在进行的合并
git reset --hard ORIG_HEAD   # 恢复到合并前的状态
```

### 修改提交历史

```bash
# 交互式变基（修改最近3次提交）
git rebase -i HEAD~3

# 在编辑器中：
# pick -> edit    # 修改提交
# pick -> squash  # 合并到上一个提交
# pick -> drop    # 删除提交
```

---

## 最佳实践

1. **提交信息规范**

   - 使用清晰、描述性的提交信息
   - 第一行简短总结（50 字符以内）
   - 详细说明在空行后

2. **提交频率**

   - 频繁提交，每次提交完成一个小功能
   - 保持提交的逻辑性

3. **分支策略**

   - 主分支（main/master）保持稳定
   - 功能开发使用独立分支
   - 使用有意义的分支名

4. **代码审查**

   - 使用 Pull Request/Merge Request
   - 在合并前进行代码审查

5. **备份**
   - 定期推送到远程仓库
   - 重要更改前创建标签

---

## 总结

本教程涵盖了 Git 的基础操作，包括：

- 仓库初始化和配置
- 基本文件操作
- 分支管理
- 远程仓库协作
- 撤销和恢复操作
- 查看历史记录
- 标签管理

继续学习建议：

- Git 工作流（Git Flow, GitHub Flow）
- 高级变基操作
- Git hooks
- 子模块和子树
- 性能优化技巧
- 团队协作最佳实践
