# Python 基础语法教程

## 目录

1. [变量和数据类型](#变量和数据类型)
2. [字符串操作](#字符串操作)
3. [运算符](#运算符)
4. [控制流](#控制流)
5. [函数](#函数)
6. [数据结构](#数据结构)
7. [模块导入](#模块导入)
8. [日期和时间处理](#日期和时间处理)
9. [文件操作](#文件操作)
10. [文件系统操作](#文件系统操作)
11. [面向对象编程](#面向对象编程)

---

## 变量和数据类型

### 变量定义

Python 中的变量不需要声明类型，直接赋值即可：

```python
# 整数
age = 25
count = 100

# 浮点数
price = 19.99
pi = 3.14159

# 字符串
name = "Python"
message = 'Hello World'

# 布尔值
is_active = True
is_complete = False

# None 类型
value = None
```

### 基本数据类型

```python
# 查看变量类型
type(age)        # <class 'int'>
type(price)      # <class 'float'>
type(name)       # <class 'str'>
type(is_active)  # <class 'bool'>
```

### 类型转换

```python
# 字符串转整数
num_str = "123"
num_int = int(num_str)

# 整数转字符串
age_str = str(25)

# 字符串转浮点数
price_float = float("19.99")

# 布尔值转换
bool(1)   # True
bool(0)   # False
bool("")  # False
bool("a") # True
```

---

## 字符串操作

### 字符串格式化

```python
name = "Python"
age = 30

# 方法1: f-string (推荐，Python 3.6+)
message = f"我的名字是 {name}，今年 {age} 岁"
# "我的名字是 Python，今年 30 岁"

# 方法2: format() 方法
message = "我的名字是 {}，今年 {} 岁".format(name, age)

# 方法3: % 格式化（旧式）
message = "我的名字是 %s，今年 %d 岁" % (name, age)

# f-string 支持表达式
result = f"10 + 20 = {10 + 20}"  # "10 + 20 = 30"
```

### 常用字符串方法

```python
text = "  Hello, World!  "

# 去除首尾空白字符
text.strip()        # "Hello, World!"
text.lstrip()       # "Hello, World!  " (去除左边)
text.rstrip()       # "  Hello, World!" (去除右边)

# 替换字符
text.replace("World", "Python")  # "  Hello, Python!  "
text.replace("l", "L", 1)        # "  HeLlo, World!  " (只替换第一个)

# 大小写转换
"hello".upper()     # "HELLO"
"HELLO".lower()     # "hello"
"hello world".title()  # "Hello World"

# 查找和检查
"hello".startswith("he")  # True
"hello".endswith("lo")   # True
"hello".find("ll")        # 2 (返回索引，找不到返回 -1)
"hello".count("l")        # 2 (统计出现次数)

# 分割和连接
"apple,banana,orange".split(",")  # ["apple", "banana", "orange"]
" ".join(["apple", "banana"])     # "apple banana"
"\n".join(["line1", "line2"])     # "line1\nline2"

# 长度
len("hello")  # 5
```

### 字符串切片

```python
text = "Python"

text[0]      # "P" (第一个字符)
text[-1]     # "n" (最后一个字符)
text[0:3]    # "Pyt" (切片，从索引0到3，不包含3)
text[:3]     # "Pyt" (从开头到索引3)
text[3:]     # "hon" (从索引3到结尾)
text[::2]    # "Pto" (步长为2，每隔一个字符)
```

---

## 运算符

### 算术运算符

```python
a = 10
b = 3

a + b   # 13 (加法)
a - b   # 7  (减法)
a * b   # 30 (乘法)
a / b   # 3.333... (除法)
a // b  # 3  (整除)
a % b   # 1  (取余)
a ** b  # 1000 (幂运算)
```

### 比较运算符

```python
a == b  # False (等于)
a != b  # True  (不等于)
a > b   # True  (大于)
a < b   # False (小于)
a >= b  # True  (大于等于)
a <= b  # False (小于等于)
```

### 逻辑运算符

```python
x = True
y = False

x and y  # False (与)
x or y   # True  (或)
not x    # False (非)
```

### 赋值运算符

```python
a = 10
a += 5   # a = 15 (等同于 a = a + 5)
a -= 3   # a = 12
a *= 2   # a = 24
a /= 4   # a = 6.0
```

---

## 控制流

### if-elif-else 语句

```python
score = 85

if score >= 90:
    grade = "A"
elif score >= 80:
    grade = "B"
elif score >= 70:
    grade = "C"
else:
    grade = "D"

print(f"成绩等级: {grade}")
```

### for 循环

```python
# 遍历列表
fruits = ["apple", "banana", "orange"]
for fruit in fruits:
    print(fruit)

# 使用 range
for i in range(5):
    print(i)  # 0, 1, 2, 3, 4

# 带索引遍历
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

### while 循环

```python
count = 0
while count < 5:
    print(count)
    count += 1

# 无限循环（需要 break 退出）
while True:
    user_input = input("输入 'quit' 退出: ")
    if user_input == "quit":
        break
```

### break 和 continue

```python
# break: 跳出循环
for i in range(10):
    if i == 5:
        break
    print(i)  # 0, 1, 2, 3, 4

# continue: 跳过当前迭代
for i in range(10):
    if i % 2 == 0:
        continue
    print(i)  # 1, 3, 5, 7, 9
```

---

## 函数

### 函数定义

```python
def greet(name):
    """问候函数"""
    return f"Hello, {name}!"

result = greet("Python")
print(result)  # Hello, Python!
```

### 默认参数

```python
def power(base, exponent=2):
    """计算幂，默认指数为 2"""
    return base ** exponent

power(3)      # 9 (3^2)
power(3, 3)   # 27 (3^3)
```

### 关键字参数

```python
def introduce(name, age, city):
    return f"{name}, {age}岁, 来自{city}"

# 使用关键字参数
introduce(name="张三", age=25, city="北京")
introduce(city="上海", name="李四", age=30)
```

### 可变参数

```python
# *args: 接收任意数量的位置参数
def sum_all(*args):
    total = 0
    for num in args:
        total += num
    return total

sum_all(1, 2, 3, 4, 5)  # 15

# **kwargs: 接收任意数量的关键字参数
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Python", version=3.9, year=2020)
```

### lambda 函数

```python
# 匿名函数
square = lambda x: x ** 2
square(5)  # 25

# 在列表中使用
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
# [1, 4, 9, 16, 25]
```

---

## 数据结构

### 列表 (List)

```python
# 创建列表
fruits = ["apple", "banana", "orange"]
numbers = [1, 2, 3, 4, 5]

# 访问元素
fruits[0]      # "apple"
fruits[-1]     # "orange" (最后一个元素)

# 切片
numbers[1:3]   # [2, 3]
numbers[:3]    # [1, 2, 3]
numbers[2:]    # [3, 4, 5]

# 添加元素
fruits.append("grape")
fruits.insert(1, "mango")

# 删除元素
fruits.remove("banana")
fruits.pop()   # 删除并返回最后一个元素
del fruits[0]

# 列表方法
len(fruits)           # 长度
fruits.count("apple") # 计数
fruits.index("orange") # 索引
fruits.sort()         # 排序
fruits.reverse()      # 反转
```

### 字典 (Dictionary)

```python
# 创建字典
student = {
    "name": "张三",
    "age": 20,
    "grade": "A"
}

# 访问元素
student["name"]        # "张三"
student.get("age")     # 20
student.get("city", "未知")  # 如果不存在返回默认值

# 添加/修改元素
student["city"] = "北京"
student["age"] = 21

# 删除元素
del student["grade"]
city = student.pop("city")

# 字典方法
student.keys()    # 所有键
student.values()  # 所有值
student.items()   # 所有键值对

# 遍历字典
for key, value in student.items():
    print(f"{key}: {value}")
```

### 元组 (Tuple)

```python
# 创建元组（不可变）
coordinates = (10, 20)
point = 30, 40  # 也可以不加括号

# 访问元素
coordinates[0]  # 10
coordinates[1]  # 20

# 元组解包
x, y = coordinates
print(x, y)  # 10 20

# 元组不可修改
# coordinates[0] = 15  # 错误！
```

### 集合 (Set)

```python
# 创建集合（无序、不重复）
numbers = {1, 2, 3, 4, 5}
colors = set(["red", "green", "blue"])

# 添加元素
numbers.add(6)
numbers.update([7, 8, 9])

# 删除元素
numbers.remove(5)
numbers.discard(10)  # 如果不存在不会报错

# 集合运算
a = {1, 2, 3, 4}
b = {3, 4, 5, 6}

a | b  # 并集 {1, 2, 3, 4, 5, 6}
a & b  # 交集 {3, 4}
a - b  # 差集 {1, 2}
a ^ b  # 对称差集 {1, 2, 5, 6}
```

### 列表推导式

```python
# 基本语法
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 带条件
even_squares = [x**2 for x in range(10) if x % 2 == 0]
# [0, 4, 16, 36, 64]

# 字典推导式
square_dict = {x: x**2 for x in range(5)}
# {0: 0, 1: 1, 2: 4, 3: 9, 4: 16}
```

---

## 模块导入

### import 语句

```python
# 导入整个模块
import os
import datetime

# 使用模块中的函数
os.path.exists("file.txt")
datetime.datetime.now()

# 从模块中导入特定函数/类
from datetime import datetime, timedelta

# 直接使用，不需要模块名前缀
now = datetime.now()
week_ago = now - timedelta(days=7)

# 导入并重命名
import feedparser as fp
feed = fp.parse(url)

# 导入多个
from datetime import datetime, timedelta, date
```

### 常用标准库模块

```python
# 操作系统相关
import os          # 文件和目录操作
import sys         # 系统相关参数和函数

# 日期时间
from datetime import datetime, timedelta, date

# 文件路径
import os.path     # 路径操作

# 正则表达式
import re          # 正则表达式匹配

# JSON 处理
import json        # JSON 编码解码

# 网络请求（需要安装）
import requests    # HTTP 库
```

---

## 日期和时间处理

### datetime 模块

```python
from datetime import datetime, timedelta, date

# 获取当前时间
now = datetime.now()           # 2025-01-09 14:30:45.123456
today = date.today()           # 2025-01-09 (只有日期)

# 创建特定日期时间
dt = datetime(2025, 1, 9, 14, 30, 0)
d = date(2025, 1, 9)

# 日期运算
today = date.today()
yesterday = today - timedelta(days=1)      # 昨天
week_ago = today - timedelta(days=7)      # 一周前
next_week = today + timedelta(days=7)     # 一周后

# 日期比较
date1 = date(2025, 1, 1)
date2 = date(2025, 1, 9)
date1 < date2      # True
date1 <= date2     # True

# 格式化日期字符串
dt = datetime.now()
dt.strftime("%Y-%m-%d %H:%M:%S")  # "2025-01-09 14:30:45"
dt.strftime("%Y年%m月%d日")        # "2025年01月09日"

# 从字符串解析日期
date_str = "2025-01-09"
d = datetime.strptime(date_str, "%Y-%m-%d").date()

# 日期属性
dt.year      # 2025
dt.month     # 1
dt.day       # 9
dt.hour      # 14
dt.minute    # 30
```

### 常用日期格式

```python
# %Y - 四位数年份 (2025)
# %m - 月份 (01-12)
# %d - 日期 (01-31)
# %H - 小时 (00-23)
# %M - 分钟 (00-59)
# %S - 秒 (00-59)
```

---

## 文件操作

### 读取文件

```python
# 读取整个文件
with open("file.txt", "r", encoding="utf-8") as f:
    content = f.read()

# 逐行读取
with open("file.txt", "r", encoding="utf-8") as f:
    for line in f:
        print(line.strip())

# 读取所有行到列表
with open("file.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()
```

### 写入文件

```python
# 写入文件（覆盖）
with open("output.txt", "w", encoding="utf-8") as f:
    f.write("Hello, World!\n")
    f.write("这是第二行\n")

# 追加模式
with open("output.txt", "a", encoding="utf-8") as f:
    f.write("追加的内容\n")
```

### 文件模式

```python
# "r"  - 只读模式（默认）
# "w"  - 写入模式（覆盖）
# "a"  - 追加模式
# "x"  - 创建模式（文件不存在时创建）
# "b"  - 二进制模式（如 "rb", "wb"）
# "t"  - 文本模式（默认）
# "+"  - 读写模式（如 "r+", "w+"）
```

---

## 文件系统操作

### os 模块常用操作

```python
import os

# 检查文件/文件夹是否存在
os.path.exists("file.txt")        # True/False
os.path.isfile("file.txt")        # 是否为文件
os.path.isdir("folder")           # 是否为文件夹

# 创建文件夹
os.makedirs("data")               # 创建单层文件夹
os.makedirs("data/sub", exist_ok=True)  # 创建多层，exist_ok=True 表示已存在不报错

# 路径操作
os.path.join("data", "file.txt")  # "data/file.txt" (跨平台路径拼接)
os.path.dirname("/path/to/file.txt")  # "/path/to" (获取目录)
os.path.basename("/path/to/file.txt")  # "file.txt" (获取文件名)
os.path.split("/path/to/file.txt")     # ("/path/to", "file.txt")

# 获取当前工作目录
current_dir = os.getcwd()

# 列出目录内容
files = os.listdir(".")           # 当前目录所有文件和文件夹

# 删除文件/文件夹
os.remove("file.txt")             # 删除文件
os.rmdir("empty_folder")          # 删除空文件夹
```

### 路径操作示例

```python
import os

# 创建 data 文件夹（如果不存在）
data_dir = "data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
    print(f"创建文件夹: {data_dir}")

# 组合文件路径（推荐使用 os.path.join）
filename = "news.txt"
filepath = os.path.join(data_dir, filename)
# Windows: "data\news.txt"
# Linux/Mac: "data/news.txt"

# 写入文件
with open(filepath, "w", encoding="utf-8") as f:
    f.write("内容")
```

---

## 面向对象编程

### 类和对象

面向对象编程（OOP）是一种编程范式，通过类和对象来组织代码。类是一个模板，对象是类的实例。

```python
# 定义一个简单的类
class Person:
    """人类"""

    # __init__ 是构造函数，创建对象时自动调用
    def __init__(self, name, age):
        self.name = name  # self 代表对象本身
        self.age = age

    # 定义方法（函数）
    def introduce(self):
        """自我介绍"""
        return f"我是{self.name}，今年{self.age}岁"

    def have_birthday(self):
        """过生日，年龄加1"""
        self.age += 1
        return f"{self.name}过生日了，现在{self.age}岁"

# 创建对象（实例化）
person1 = Person("张三", 25)
person2 = Person("李四", 30)

# 访问属性
print(person1.name)  # "张三"
print(person2.age)   # 30

# 调用方法
print(person1.introduce())  # "我是张三，今年25岁"
person1.have_birthday()     # "张三过生日了，现在26岁"
```

### 类属性和实例属性

```python
class Student:
    # 类属性（所有实例共享）
    school = "Python学校"

    def __init__(self, name, grade):
        # 实例属性（每个对象独有）
        self.name = name
        self.grade = grade

    def info(self):
        return f"{self.name}在{self.school}，{self.grade}年级"

student1 = Student("小明", 1)
student2 = Student("小红", 2)

print(student1.school)  # "Python学校"
print(student2.school)  # "Python学校"

# 修改类属性会影响所有实例
Student.school = "新学校"
print(student1.school)  # "新学校"
```

### 继承

继承允许一个类（子类）继承另一个类（父类）的属性和方法。

```python
# 父类（基类）
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return f"{self.name}在叫"

# 子类（派生类）
class Dog(Animal):
    def speak(self):
        # 重写父类方法
        return f"{self.name}在汪汪叫"

    def fetch(self):
        # 子类特有的方法
        return f"{self.name}去捡球了"

class Cat(Animal):
    def speak(self):
        return f"{self.name}在喵喵叫"

# 使用继承
dog = Dog("旺财")
cat = Cat("小花")

print(dog.speak())  # "旺财在汪汪叫"
print(cat.speak())  # "小花在喵喵叫"
print(dog.fetch())  # "旺财去捡球了"
```

### 实际应用示例

```python
class NewsArticle:
    """新闻文章类"""

    def __init__(self, title, link, content=""):
        self.title = title
        self.link = link
        self.content = content

    def get_summary(self, max_length=100):
        """获取摘要"""
        if len(self.content) <= max_length:
            return self.content
        return self.content[:max_length] + "..."

    def save_to_file(self, filename):
        """保存到文件"""
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"标题: {self.title}\n")
            f.write(f"链接: {self.link}\n")
            f.write(f"\n内容:\n{self.content}")

# 使用类
article = NewsArticle(
    title="Python 学习指南",
    link="https://example.com",
    content="这是一篇关于Python学习的文章..."
)

print(article.get_summary(50))  # "这是一篇关于Python学习的文章..."
article.save_to_file("news.txt")
```

---

## 异常处理

```python
try:
    result = 10 / 0
except ZeroDivisionError:
    print("不能除以零！")
except Exception as e:
    print(f"发生错误: {e}")
else:
    print("没有错误发生")
finally:
    print("无论是否出错都会执行")
```

---

## 常用内置函数

```python
# 类型转换
int(), float(), str(), bool(), list(), dict(), tuple(), set()

# 数学函数
abs(-5)        # 5 (绝对值)
max(1, 2, 3)   # 3
min(1, 2, 3)   # 1
sum([1, 2, 3]) # 6
round(3.14159, 2)  # 3.14

# 序列函数
len([1, 2, 3])     # 3
sorted([3, 1, 2])  # [1, 2, 3]
reversed([1, 2, 3]) # 反转迭代器

# 其他
range(5)           # 0, 1, 2, 3, 4
enumerate(["a", "b"])  # (0, 'a'), (1, 'b')
zip([1, 2], ["a", "b"])  # (1, 'a'), (2, 'b')

# 对象属性检查
hasattr(obj, "attribute")  # 检查对象是否有某个属性
getattr(obj, "name", "default")  # 获取属性，不存在返回默认值
```

### 对象属性操作

```python
# hasattr() - 检查对象是否有某个属性
class Person:
    def __init__(self, name):
        self.name = name

person = Person("张三")
hasattr(person, "name")    # True
hasattr(person, "age")     # False

# getattr() - 获取属性值，可设置默认值
name = getattr(person, "name", "未知")      # "张三"
age = getattr(person, "age", 0)             # 0 (默认值)

# 访问对象属性
person.name              # "张三" (直接访问)
person.__dict__         # {"name": "张三"} (所有属性)
```

---

## 总结

本教程涵盖了 Python 的基础语法，包括：

- 变量和数据类型
- 字符串操作（格式化、常用方法）
- 运算符和控制流
- 函数定义和使用
- 常用数据结构
- 模块导入
- 日期和时间处理
- 文件操作和文件系统操作
- 面向对象编程（类、对象、继承）
- 异常处理

### 项目实战要点

在 `news.py` 项目中，我们实际使用了以下知识点：

1. **模块导入**：

   - `import feedparser` - 解析 RSS 订阅源
   - `from bs4 import BeautifulSoup` - 解析 HTML
   - `import os` - 文件系统操作
   - `import requests` - 发送 HTTP 请求

2. **字符串操作**：

   - `str.replace()` - 替换字符（清理文件名）
   - `str.strip()` - 去除首尾空白
   - `str.join()` - 连接列表元素
   - `len()` - 获取字符串长度
   - 字符串切片 `[:50]` - 截取前 50 个字符
   - f-string 格式化 - `f"找到 {len(feed.entries)} 条新闻"`

3. **列表操作**：

   - `list.append()` - 添加元素
   - 列表切片 `[:10]` - 取前 10 条新闻
   - `for` 循环遍历列表

4. **文件系统**：

   - `os.path.exists()` - 检查文件夹是否存在
   - `os.makedirs()` - 创建文件夹
   - `os.path.join()` - 组合文件路径

5. **对象属性访问**：

   - `entry.title` - 访问新闻标题
   - `entry.link` - 访问新闻链接
   - `feed.entries` - 访问新闻列表

6. **异常处理**：

   - `try/except` - 捕获网络请求错误
   - `continue` - 跳过当前循环迭代

7. **文件操作**：

   - `with open()` - 打开文件（自动关闭）
   - `f.write()` - 写入文件内容

8. **函数定义**：

   - `def clean_filename()` - 定义清理文件名的函数

9. **面向对象编程**：
   - 使用类来组织代码，提高代码的可维护性和复用性
   - 可以将 `news.py` 中的功能封装成类，例如 `NewsCrawler` 类

继续学习建议：

- 面向对象高级特性（多态、封装、特殊方法）
- 正则表达式（re 模块）
- 网络请求（requests 库）
- HTML 解析（BeautifulSoup）
- 数据库操作
- Web 开发框架
