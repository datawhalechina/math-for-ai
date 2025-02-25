## 习题

### 5.1 
计算$f(x)=\log(x^{4})\sin(x^{3})$的导数$f'(x)$。

### 5.2 
计算 Logistic 函数$\displaystyle f(x)=\frac{1}{1 + \exp(-x)}$的导数$f'(x)$。

### 5.3
计算函数$\displaystyle f(x)=\exp\left[ -\frac{1}{2\sigma^{2}}(x - \mu)^{2} \right]$的导数$f'(x)$，其中$\mu,\sigma\in \mathbb{R}$是常数。

### 5.4
计算$f(x)=\sin(x)+\cos(x)$在$x_{0}=0$处的 Taylor 多项式$T_{n}$，$n = 0,\cdots,5$。

### 5.5
考虑以下函数：
$$
\begin{align}
f_{1}(x)&=\sin(x_{1})\cos(x_{2}), &x\in \mathbb{R}^{2} \\
f_{2}(x,y)&=x^{\top}y, &x,y\in \mathbb{R}^{n}\\
f_{3}(x)&=x x^{\top}, &x\in \mathbb{R}^{n}
\end{align}
$$
（a）$\displaystyle \frac{\partial f_{i}}{\partial x}$的维度是多少？

（b）计算 Jacobi 矩阵。

### 5.6
对$f$关于$t$求导，对$g$关于$x$求导，其中$f(t)=\sin(\log(t^{\top}t))$，$t\in \mathbb{R}^{D}$，$g(X)=\text{tr}(AXB)$，$A\in \mathbb{R}^{D\times E}$，$X\in \mathbb{R}^{E\times F}$，$B\in \mathbb{R}^{F\times D}$，其中$\text{tr}(\cdot)$表示矩阵的迹。

### 5.7
使用链式法则计算以下函数的导数$\displaystyle \frac{\mathrm{d}f}{\mathrm{d}x}$。并写出每个偏导数的维度。
（a）$f(z)=\log(1 + z)$，$z = x^{\top}x$，$x\in \mathbb{R}^{D}$

（b）$f(z)=\sin(z)$，$z = Ax + b$，$A\in \mathbb{R}^{E\times D}$，$x\in \mathbb{R}^{D}$，$b\in \mathbb{R}^{E}$，其中 $\sin(\cdot)$ 作用于 $z$ 的每个分量。

### 5.8
计算以下函数的导数$\displaystyle \frac{\mathrm{d}f}{\mathrm{d}x}$。
（a）使用链式法则，并给出每个一阶偏导数的维度$$\begin{align}f(z)&=\exp(-\frac{1}{2}z)\\ z &= g(\boldsymbol{ y } )=\boldsymbol{ y } ^{\top}\boldsymbol{ S } ^{-1}\boldsymbol{ y }  \\ \boldsymbol{ y }  &= h(\boldsymbol{ x } )=\boldsymbol{ x }  - \boldsymbol{ \mu } \end{align}$$其中 $\boldsymbol{ x },\boldsymbol{ \mu }\in \mathbb{R}^{D}$，$\boldsymbol{ S }\in \mathbb{R}^{D\times D}$。

（b）$$f(x)=tr(xx^{\top}+\sigma^{2}I),$$其中 $x\in \mathbb{R}^{D}$。这里$\text{tr}(A)$是$A$的迹，即对角元素$A_{ii}$的和。(提示：显式写出外积)

（c）使用链式法则，给出每个一阶偏导数的维度（不需要显式计算偏导数的乘积）。$$\begin{align}f &= \tanh(z)\in \mathbb{R}^{M} \\ z &= Ax + b, &x\in \mathbb{R}^{N}, A\in \mathbb{R}^{M\times N}, b\in \mathbb{R}^{M}.\end{align}$$这里，$\tanh$应用于 $z$ 的每个分量。

### 5.9
我们定义$$\begin{align}g(z,\nu)&:=\log p(\boldsymbol{ x } ,\boldsymbol{ z } )-\log q(\boldsymbol{ z } ,\boldsymbol{ \nu } )\\\boldsymbol{ z }  &:= t(\boldsymbol{ \epsilon } ,\boldsymbol{ \nu } )\end{align}$$对于可微函数$p,q,t$以及$\boldsymbol{ x }\in \mathbb{R}^{D}$，$\boldsymbol{ z }\in \mathbb{R}^{E}$，$\boldsymbol{ \nu } \in \mathbb{R}^{F}$，$\boldsymbol{ \epsilon } \in \mathbb{R}^{G}$。使用链式法则计算梯度$$\frac{\mathrm{d}}{\mathrm{d}\boldsymbol{ \nu } }g(\boldsymbol{ z } ,\boldsymbol{ \nu } ).$$
