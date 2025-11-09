# Linux 基础语法教程

## 目录

1. [文件系统基础](#文件系统基础)
2. [基本命令](#基本命令)
3. [文件操作](#文件操作)
4. [权限管理](#权限管理)
5. [文本处理](#文本处理)
6. [进程管理](#进程管理)
7. [网络命令](#网络命令)
8. [管道和重定向](#管道和重定向)

---

## 文件系统基础

### 目录结构

```bash
/              # 根目录
/bin           # 基本命令
/etc           # 配置文件
/home          # 用户主目录
/usr           # 用户程序
/var           # 可变数据
/tmp           # 临时文件
/opt           # 可选软件
```

### 路径表示

```bash
# 绝对路径（从根目录开始）
/home/user/documents/file.txt

# 相对路径（从当前目录开始）
./file.txt          # 当前目录
../parent/file.txt  # 上级目录
~/documents/file.txt # 用户主目录
```

---

## 基本命令

### 查看当前目录

```bash
pwd  # 显示当前工作目录
```

### 切换目录

```bash
cd /home/user        # 切换到绝对路径
cd documents         # 切换到相对路径
cd ..                # 返回上级目录
cd ~                 # 返回用户主目录
cd -                 # 返回上一个目录
cd                   # 返回用户主目录
```

### 列出文件

```bash
ls                  # 列出当前目录文件
ls -l               # 详细列表（长格式）
ls -a               # 显示所有文件（包括隐藏文件）
ls -h               # 人类可读的文件大小
ls -t               # 按修改时间排序
ls -r               # 反向排序
ls -la              # 组合使用：详细列表+显示隐藏文件
ls /home/user       # 列出指定目录
```

### 创建和删除目录

```bash
mkdir dirname              # 创建目录
mkdir -p dir1/dir2/dir3    # 创建多级目录
rmdir dirname              # 删除空目录
rm -r dirname              # 递归删除目录（包括内容）
rm -rf dirname             # 强制递归删除（危险！）
```

---

## 文件操作

### 创建文件

```bash
touch filename.txt         # 创建空文件或更新时间戳
echo "内容" > file.txt     # 创建文件并写入内容
cat > file.txt             # 从键盘输入创建文件（Ctrl+D 结束）
```

### 查看文件内容

```bash
cat file.txt               # 显示整个文件
less file.txt              # 分页显示（空格翻页，q退出）
more file.txt              # 分页显示（较老的命令）
head -n 10 file.txt        # 显示前10行
tail -n 10 file.txt        # 显示后10行
tail -f file.txt           # 实时跟踪文件变化（日志常用）
```

### 复制和移动

```bash
cp source.txt dest.txt           # 复制文件
cp -r source_dir dest_dir        # 递归复制目录
cp file.txt /home/user/          # 复制到指定目录

mv oldname.txt newname.txt       # 重命名文件
mv file.txt /home/user/          # 移动文件
mv file1.txt file2.txt dir/      # 移动多个文件到目录
```

### 删除文件

```bash
rm file.txt              # 删除文件
rm -i file.txt           # 交互式删除（确认）
rm -f file.txt           # 强制删除（不提示）
rm *.txt                 # 删除所有 .txt 文件
```

### 查找文件

```bash
find /home -name "file.txt"           # 按名称查找
find /home -type f -name "*.txt"      # 查找所有 .txt 文件
find /home -size +100M                # 查找大于100MB的文件
find /home -mtime -7                  # 查找7天内修改的文件

# locate 命令（需要先更新数据库：sudo updatedb）
locate file.txt
```

---

## 权限管理

### 查看权限

```bash
ls -l file.txt
# 输出示例：-rw-r--r-- 1 user group 1024 Jan 1 10:00 file.txt
# 权限位：-rw-r--r--
# -: 文件类型（- 普通文件，d 目录，l 链接）
# rw-: 所有者权限（读、写、执行）
# r--: 组权限
# r--: 其他用户权限
```

### 修改权限（chmod）

```bash
# 数字方式
chmod 755 file.txt        # rwxr-xr-x
chmod 644 file.txt        # rw-r--r--
chmod 777 file.txt        # rwxrwxrwx（所有权限）

# 符号方式
chmod u+x file.txt        # 给所有者添加执行权限
chmod g-w file.txt        # 移除组的写权限
chmod o+r file.txt        # 给其他用户添加读权限
chmod a+x file.txt        # 给所有人添加执行权限
chmod u=rwx,g=rx,o=r file.txt  # 分别设置权限

# 递归修改目录
chmod -R 755 directory/
```

### 权限数字含义

```
7 = 4+2+1 = rwx (读+写+执行)
6 = 4+2   = rw- (读+写)
5 = 4+1   = r-x (读+执行)
4 = 4     = r-- (只读)
0 = 0     = --- (无权限)
```

### 修改所有者（chown）

```bash
chown user file.txt              # 修改所有者
chown user:group file.txt        # 修改所有者和组
chown -R user:group directory/   # 递归修改
```

### 修改组（chgrp）

```bash
chgrp groupname file.txt
chgrp -R groupname directory/
```

---

## 文本处理

### grep - 搜索文本

```bash
grep "pattern" file.txt          # 搜索包含 pattern 的行
grep -i "pattern" file.txt       # 忽略大小写
grep -n "pattern" file.txt       # 显示行号
grep -v "pattern" file.txt       # 显示不匹配的行
grep -r "pattern" directory/     # 递归搜索
grep -E "pattern1|pattern2" file.txt  # 正则表达式
```

### sed - 流编辑器

```bash
sed 's/old/new/g' file.txt       # 替换所有 old 为 new
sed 's/old/new/' file.txt        # 只替换每行第一个
sed -i 's/old/new/g' file.txt    # 直接修改文件
sed '2d' file.txt                # 删除第2行
sed '2,5d' file.txt              # 删除2-5行
sed '2a\新行内容' file.txt        # 在第2行后添加
```

### awk - 文本处理工具

```bash
awk '{print $1}' file.txt        # 打印第一列
awk '{print $1, $3}' file.txt    # 打印第1和第3列
awk -F: '{print $1}' /etc/passwd # 使用冒号作为分隔符
awk '/pattern/ {print $0}' file.txt  # 匹配 pattern 的行
awk '{sum+=$1} END {print sum}' file.txt  # 计算第一列总和
```

### sort - 排序

```bash
sort file.txt                    # 按字母顺序排序
sort -n file.txt                 # 按数字排序
sort -r file.txt                 # 反向排序
sort -u file.txt                 # 去重排序
sort -k 2 file.txt              # 按第2列排序
```

### uniq - 去重

```bash
uniq file.txt                    # 去除连续重复行
uniq -c file.txt                 # 显示重复次数
uniq -d file.txt                 # 只显示重复行
uniq -u file.txt                 # 只显示唯一行
```

### cut - 截取列

```bash
cut -d: -f1 /etc/passwd          # 以冒号分隔，取第1列
cut -c1-5 file.txt               # 取第1-5个字符
cut -f1,3 file.txt               # 取第1和第3列（默认制表符分隔）
```

---

## 进程管理

### 查看进程

```bash
ps                    # 显示当前终端进程
ps aux                # 显示所有进程
ps -ef                # 另一种格式
top                   # 实时显示进程（q退出）
htop                  # 更友好的进程查看器（需安装）
```

### 进程控制

```bash
# 后台运行
command &             # 在后台运行命令

# 查看后台任务
jobs                  # 显示后台任务列表
fg %1                 # 将任务1调到前台
bg %1                 # 将任务1放到后台运行

# 终止进程
kill PID              # 终止进程（PID是进程ID）
kill -9 PID           # 强制终止
killall process_name # 终止所有同名进程
pkill process_name    # 按名称终止进程
```

### 查找进程

```bash
ps aux | grep process_name       # 查找进程
pgrep process_name               # 查找进程ID
pidof process_name               # 查找进程ID
```

---

## 网络命令

### 网络连接

```bash
ping google.com                  # 测试网络连接
ping -c 4 google.com            # 发送4个包后停止

# 查看网络配置
ifconfig                         # 查看网络接口（旧命令）
ip addr                          # 查看IP地址（新命令）
ip link                          # 查看网络接口
```

### 端口和连接

```bash
netstat -an                      # 显示所有网络连接
netstat -tulpn                   # 显示监听端口
ss -tulpn                        # 更现代的替代命令

# 测试端口
telnet hostname 80               # 测试TCP连接
nc -zv hostname 80               # 使用 netcat 测试
```

### 下载文件

```bash
wget http://example.com/file.zip # 下载文件
wget -O output.zip http://...    # 指定输出文件名
wget -c http://...               # 断点续传

curl http://example.com          # 下载并显示内容
curl -O http://example.com/file.zip  # 下载文件
curl -L http://...               # 跟随重定向
```

---

## 管道和重定向

### 重定向

```bash
# 输出重定向
command > file.txt               # 覆盖写入
command >> file.txt              # 追加写入

# 输入重定向
command < file.txt               # 从文件读取输入
command << EOF                   # 多行输入（直到EOF）
内容
EOF

# 错误重定向
command 2> error.log             # 错误输出到文件
command 2>&1                     # 错误输出合并到标准输出
command > output.log 2>&1        # 所有输出到文件
```

### 管道

```bash
# 管道：将一个命令的输出作为另一个命令的输入
ls -l | grep ".txt"              # 列出文件并过滤 .txt
ps aux | grep python             # 查找 Python 进程
cat file.txt | sort | uniq       # 排序并去重

# 常用管道组合
ls -l | head -10                 # 显示前10个文件
cat file.txt | wc -l             # 统计行数
cat file.txt | grep "pattern" | sort  # 搜索、排序
```

---

## 压缩和解压

### tar

```bash
# 创建压缩包
tar -czf archive.tar.gz directory/    # gzip 压缩
tar -cjf archive.tar.bz2 directory/   # bzip2 压缩
tar -cJf archive.tar.xz directory/    # xz 压缩

# 解压
tar -xzf archive.tar.gz               # 解压 gzip
tar -xjf archive.tar.bz2               # 解压 bzip2
tar -xJf archive.tar.xz                # 解压 xz

# 查看压缩包内容
tar -tzf archive.tar.gz                # 列出内容
```

### zip/unzip

```bash
zip -r archive.zip directory/          # 创建 zip 压缩包
unzip archive.zip                      # 解压 zip
unzip -l archive.zip                   # 查看内容
unzip -d destination archive.zip      # 解压到指定目录
```

### gzip/gunzip

```bash
gzip file.txt                          # 压缩文件（生成 file.txt.gz）
gunzip file.txt.gz                     # 解压文件
gzip -d file.txt.gz                    # 解压文件（另一种方式）
```

---

## 系统信息

### 查看系统信息

```bash
uname -a                               # 系统信息
hostname                               # 主机名
uptime                                 # 系统运行时间
whoami                                 # 当前用户
id                                     # 用户ID信息
```

### 磁盘使用

```bash
df -h                                  # 磁盘使用情况（人类可读）
du -h directory/                       # 目录大小
du -sh *                               # 当前目录各文件/目录大小
du -h --max-depth=1                    # 只显示一级目录
```

### 内存使用

```bash
free -h                                # 内存使用情况
free -m                                # 以MB显示
```

---

## 环境变量

```bash
echo $PATH                             # 查看 PATH 变量
export VAR="value"                      # 设置环境变量
export PATH=$PATH:/new/path            # 添加路径到 PATH

# 查看所有环境变量
env
printenv

# 在配置文件中设置（永久生效）
# ~/.bashrc 或 ~/.zshrc
export MY_VAR="value"
```

---

## 快捷键

```bash
Ctrl + C          # 终止当前命令
Ctrl + D          # 退出终端或结束输入
Ctrl + Z          # 暂停进程（可用 fg 恢复）
Ctrl + L          # 清屏（等同于 clear）
Ctrl + A          # 光标移到行首
Ctrl + E          # 光标移到行尾
Ctrl + U          # 删除光标前所有内容
Ctrl + K          # 删除光标后所有内容
Ctrl + R          # 搜索命令历史
Tab                # 自动补全
```

---

## 总结

本教程涵盖了 Linux 的基础命令，包括：

- 文件系统操作
- 文件权限管理
- 文本处理工具
- 进程管理
- 网络命令
- 管道和重定向

继续学习建议：

- Shell 脚本编程
- 系统服务管理（systemd）
- 用户和组管理
- 软件包管理（apt/yum）
- 日志查看和分析
- 系统监控和性能优化
