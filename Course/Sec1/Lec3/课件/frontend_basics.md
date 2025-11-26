# 前端三件套基础教程

## 目录

1. [HTML 基础](#html-基础)
2. [CSS 基础](#css-基础)
3. [JavaScript 基础](#javascript-基础)
4. [综合示例](#综合示例)

---

## HTML 基础

HTML（HyperText Markup Language）是网页的结构语言，用于定义网页内容。

### 基本结构

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>网页标题</title>
</head>
<body>
    <!-- 网页内容 -->
</body>
</html>
```

### 常用标签

#### 标题和段落

```html
<h1>一级标题</h1>
<h2>二级标题</h2>
<h3>三级标题</h3>

<p>这是一个段落。</p>
<p>这是另一个段落。</p>
```

#### 链接和图片

```html
<!-- 链接 -->
<a href="https://www.example.com">访问示例网站</a>
<a href="page.html">内部链接</a>

<!-- 图片 -->
<img src="image.jpg" alt="图片描述">
```

#### 列表

```html
<!-- 无序列表 -->
<ul>
    <li>项目1</li>
    <li>项目2</li>
    <li>项目3</li>
</ul>

<!-- 有序列表 -->
<ol>
    <li>第一步</li>
    <li>第二步</li>
    <li>第三步</li>
</ol>
```

#### 表单

```html
<form>
    <label for="name">姓名：</label>
    <input type="text" id="name" name="name" placeholder="请输入姓名">
    
    <label for="email">邮箱：</label>
    <input type="email" id="email" name="email">
    
    <label for="message">留言：</label>
    <textarea id="message" name="message" rows="4"></textarea>
    
    <button type="submit">提交</button>
</form>
```

#### 容器元素

```html
<!-- div：块级容器 -->
<div>
    <p>这是一个块级容器</p>
</div>

<!-- span：行内容器 -->
<p>这是<span style="color: red;">红色</span>的文字</p>

<!-- section：语义化容器 -->
<section>
    <h2>章节标题</h2>
    <p>章节内容</p>
</section>
```

### 完整示例

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>我的第一个网页</title>
</head>
<body>
    <header>
        <h1>欢迎来到我的网站</h1>
        <nav>
            <a href="#home">首页</a>
            <a href="#about">关于</a>
            <a href="#contact">联系</a>
        </nav>
    </header>
    
    <main>
        <section id="home">
            <h2>首页</h2>
            <p>这是首页内容。</p>
        </section>
        
        <section id="about">
            <h2>关于我们</h2>
            <p>这是关于我们的内容。</p>
        </section>
    </main>
    
    <footer>
        <p>版权所有 © 2025</p>
    </footer>
</body>
</html>
```

---

## CSS 基础

CSS（Cascading Style Sheets）用于美化网页，控制样式和布局。

### 三种引入方式

#### 1. 内联样式（不推荐，仅用于测试）

```html
<p style="color: red; font-size: 20px;">红色文字</p>
```

#### 2. 内部样式表

```html
<head>
    <style>
        p {
            color: blue;
            font-size: 16px;
        }
    </style>
</head>
```

#### 3. 外部样式表（推荐）

```html
<head>
    <link rel="stylesheet" href="style.css">
</head>
```

### 基本语法

```css
/* 选择器 { 属性: 值; } */
p {
    color: red;
    font-size: 16px;
    margin: 10px;
}
```

### 常用选择器

#### 元素选择器

```css
/* 选择所有 p 标签 */
p {
    color: blue;
}

/* 选择所有 h1 标签 */
h1 {
    font-size: 24px;
}
```

#### 类选择器

```css
/* HTML: <p class="highlight">文本</p> */
.highlight {
    background-color: yellow;
    font-weight: bold;
}
```

#### ID 选择器

```css
/* HTML: <div id="header">内容</div> */
#header {
    background-color: #333;
    color: white;
}
```

#### 组合选择器

```css
/* 选择所有 div 内的 p 标签 */
div p {
    color: green;
}

/* 选择 class 为 container 的元素内的所有 p */
.container p {
    margin: 10px;
}
```

### 常用样式属性

#### 文字样式

```css
p {
    color: #333333;           /* 文字颜色 */
    font-size: 16px;           /* 字体大小 */
    font-family: Arial, sans-serif;  /* 字体 */
    font-weight: bold;         /* 字体粗细 */
    text-align: center;        /* 文字对齐 */
    line-height: 1.5;         /* 行高 */
}
```

#### 背景和边框

```css
div {
    background-color: #f0f0f0;  /* 背景颜色 */
    background-image: url('bg.jpg');  /* 背景图片 */
    border: 1px solid #ccc;     /* 边框 */
    border-radius: 5px;         /* 圆角 */
}
```

#### 尺寸和间距

```css
div {
    width: 300px;              /* 宽度 */
    height: 200px;             /* 高度 */
    margin: 20px;              /* 外边距 */
    padding: 15px;             /* 内边距 */
}
```

#### 布局

```css
/* 弹性布局 */
.container {
    display: flex;
    justify-content: center;   /* 水平居中 */
    align-items: center;       /* 垂直居中 */
}

/* 网格布局 */
.grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;  /* 三列 */
    gap: 20px;                 /* 间距 */
}
```

### 完整示例

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>CSS 示例</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            color: #333;
            text-align: center;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        
        .card {
            background-color: #f9f9f9;
            padding: 15px;
            margin: 10px 0;
            border-left: 4px solid #4CAF50;
        }
        
        .button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        }
        
        .button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>CSS 示例页面</h1>
        <div class="card">
            <p>这是一个卡片样式的内容。</p>
        </div>
        <button class="button">点击按钮</button>
    </div>
</body>
</html>
```

---

## JavaScript 基础

JavaScript 用于添加交互功能，让网页"活"起来。

### 引入方式

#### 1. 内部脚本

```html
<script>
    console.log("Hello, World!");
</script>
```

#### 2. 外部脚本（推荐）

```html
<script src="script.js"></script>
```

### 基本语法

#### 变量

```javascript
// 使用 let（推荐）
let name = "张三";
let age = 25;

// 使用 const（常量）
const PI = 3.14159;

// 使用 var（旧式，不推荐）
var oldVar = "旧变量";
```

#### 数据类型

```javascript
// 字符串
let text = "Hello";

// 数字
let number = 42;
let float = 3.14;

// 布尔值
let isTrue = true;
let isFalse = false;

// 数组
let fruits = ["苹果", "香蕉", "橙子"];

// 对象
let person = {
    name: "张三",
    age: 25,
    city: "北京"
};
```

#### 函数

```javascript
// 函数定义
function greet(name) {
    return "Hello, " + name + "!";
}

// 调用函数
let message = greet("张三");
console.log(message);

// 箭头函数（ES6）
const add = (a, b) => {
    return a + b;
};

// 简化写法
const multiply = (a, b) => a * b;
```

#### 条件语句

```javascript
let age = 18;

if (age >= 18) {
    console.log("已成年");
} else {
    console.log("未成年");
}

// 三元运算符
let status = age >= 18 ? "已成年" : "未成年";
```

#### 循环

```javascript
// for 循环
for (let i = 0; i < 5; i++) {
    console.log(i);
}

// for...of 循环（遍历数组）
let fruits = ["苹果", "香蕉", "橙子"];
for (let fruit of fruits) {
    console.log(fruit);
}

// forEach 方法
fruits.forEach(function(fruit) {
    console.log(fruit);
});
```

### DOM 操作

DOM（Document Object Model）是 HTML 文档的对象模型。

#### 获取元素

```javascript
// 通过 ID
let element = document.getElementById("myId");

// 通过类名（返回数组）
let elements = document.getElementsByClassName("myClass");

// 通过标签名
let paragraphs = document.getElementsByTagName("p");

// 使用选择器（推荐）
let element = document.querySelector("#myId");
let elements = document.querySelectorAll(".myClass");
```

#### 修改内容

```javascript
// 修改文本内容
let element = document.getElementById("demo");
element.textContent = "新内容";
element.innerHTML = "<strong>加粗内容</strong>";

// 修改样式
element.style.color = "red";
element.style.fontSize = "20px";

// 添加/移除类
element.classList.add("new-class");
element.classList.remove("old-class");
element.classList.toggle("active");
```

#### 事件处理

```javascript
// 方式1：HTML 属性（不推荐）
// <button onclick="handleClick()">点击</button>

// 方式2：JavaScript 绑定（推荐）
let button = document.getElementById("myButton");
button.addEventListener("click", function() {
    alert("按钮被点击了！");
});

// 方式3：箭头函数
button.addEventListener("click", () => {
    console.log("点击事件");
});
```

#### 常用事件

```javascript
// 点击事件
element.addEventListener("click", function() {
    console.log("被点击");
});

// 鼠标悬停
element.addEventListener("mouseenter", function() {
    element.style.backgroundColor = "yellow";
});

element.addEventListener("mouseleave", function() {
    element.style.backgroundColor = "white";
});

// 输入事件
let input = document.getElementById("myInput");
input.addEventListener("input", function() {
    console.log("输入内容：" + input.value);
});

// 表单提交
let form = document.getElementById("myForm");
form.addEventListener("submit", function(event) {
    event.preventDefault(); // 阻止默认提交
    console.log("表单提交");
});
```

### 完整示例

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>JavaScript 示例</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 600px;
            margin: 50px auto;
            padding: 20px;
        }
        button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
        }
        button:hover {
            background-color: #45a049;
        }
        #output {
            margin-top: 20px;
            padding: 10px;
            background-color: #f0f0f0;
            border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1>JavaScript 交互示例</h1>
    
    <button id="btn1">点击我</button>
    <button id="btn2">改变颜色</button>
    <button id="btn3">显示时间</button>
    
    <div id="output">等待操作...</div>
    
    <script>
        // 获取元素
        const btn1 = document.getElementById("btn1");
        const btn2 = document.getElementById("btn2");
        const btn3 = document.getElementById("btn3");
        const output = document.getElementById("output");
        
        // 按钮1：点击计数
        let count = 0;
        btn1.addEventListener("click", function() {
            count++;
            output.textContent = `按钮被点击了 ${count} 次`;
        });
        
        // 按钮2：改变背景颜色
        btn2.addEventListener("click", function() {
            const colors = ["#ff9999", "#99ff99", "#9999ff", "#ffff99"];
            const randomColor = colors[Math.floor(Math.random() * colors.length)];
            document.body.style.backgroundColor = randomColor;
            output.textContent = "背景颜色已改变！";
        });
        
        // 按钮3：显示当前时间
        btn3.addEventListener("click", function() {
            const now = new Date();
            output.textContent = "当前时间：" + now.toLocaleString();
        });
    </script>
</body>
</html>
```

---

## 综合示例

### 简单的待办事项应用

```html
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <title>待办事项</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: Arial, sans-serif;
            background-color: #f5f5f5;
            padding: 20px;
        }
        
        .container {
            max-width: 600px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        
        #todoInput {
            flex: 1;
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        
        #addBtn {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        
        #addBtn:hover {
            background-color: #45a049;
        }
        
        #todoList {
            list-style: none;
        }
        
        .todo-item {
            display: flex;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            background-color: #f9f9f9;
            border-radius: 5px;
        }
        
        .todo-item.completed {
            text-decoration: line-through;
            opacity: 0.6;
        }
        
        .todo-item input[type="checkbox"] {
            margin-right: 10px;
        }
        
        .todo-item span {
            flex: 1;
        }
        
        .delete-btn {
            background-color: #f44336;
            color: white;
            border: none;
            padding: 5px 10px;
            border-radius: 3px;
            cursor: pointer;
        }
        
        .delete-btn:hover {
            background-color: #da190b;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>📝 待办事项</h1>
        
        <div class="input-group">
            <input type="text" id="todoInput" placeholder="输入待办事项...">
            <button id="addBtn">添加</button>
        </div>
        
        <ul id="todoList"></ul>
    </div>
    
    <script>
        // 获取元素
        const todoInput = document.getElementById("todoInput");
        const addBtn = document.getElementById("addBtn");
        const todoList = document.getElementById("todoList");
        
        // 添加待办事项
        function addTodo() {
            const text = todoInput.value.trim();
            if (text === "") {
                alert("请输入待办事项！");
                return;
            }
            
            // 创建列表项
            const li = document.createElement("li");
            li.className = "todo-item";
            
            // 创建复选框
            const checkbox = document.createElement("input");
            checkbox.type = "checkbox";
            checkbox.addEventListener("change", function() {
                li.classList.toggle("completed");
            });
            
            // 创建文本
            const span = document.createElement("span");
            span.textContent = text;
            
            // 创建删除按钮
            const deleteBtn = document.createElement("button");
            deleteBtn.className = "delete-btn";
            deleteBtn.textContent = "删除";
            deleteBtn.addEventListener("click", function() {
                li.remove();
            });
            
            // 组装元素
            li.appendChild(checkbox);
            li.appendChild(span);
            li.appendChild(deleteBtn);
            
            // 添加到列表
            todoList.appendChild(li);
            
            // 清空输入框
            todoInput.value = "";
        }
        
        // 按钮点击事件
        addBtn.addEventListener("click", addTodo);
        
        // 回车键添加
        todoInput.addEventListener("keypress", function(event) {
            if (event.key === "Enter") {
                addTodo();
            }
        });
    </script>
</body>
</html>
```

---

## 总结

### HTML、CSS、JavaScript 的关系

- **HTML**：网页的骨架（结构）
- **CSS**：网页的外观（样式）
- **JavaScript**：网页的行为（交互）

### 学习路径

1. **HTML**：掌握基本标签和结构
2. **CSS**：学习选择器和常用样式
3. **JavaScript**：理解变量、函数、DOM 操作
4. **综合应用**：结合三者创建交互式网页

### 继续学习

- 响应式设计（媒体查询）
- CSS 框架（Bootstrap、Tailwind）
- JavaScript 框架（Vue、React）
- 前端工具（npm、webpack）

### 实践建议

1. 从简单的静态页面开始
2. 逐步添加样式和交互
3. 参考优秀网站的设计
4. 多动手实践，多写代码

