## 练习

5.1 计算$f(x)=\log(x^{4})\sin(x^{3})$的导数$f'(x)$。
5.2 计算 Logistic 函数$f(x)=\frac{1}{1 + \exp(-x)}$的导数$f'(x)$。
5.3计算函数$f(x)=\exp(-\frac{1}{2\sigma^{2}}(x - \mu)^{2})$的导数$f'(x)$，其中$\mu,\sigma\in \mathbb{R}$是常数。
5.4计算$f(x)=\sin(x)+\cos(x)$在$x_{0}=0$处的 Taylor 多项式$T_{n}$，$n = 0,\cdots,5$。
5.5考虑以下函数：
$f_{1}(x)=\sin(x_{1})\cos(x_{2})$，$x\in \mathbb{R}^{2}$
$f_{2}(x,y)=x^{\top}y$，$x,y\in \mathbb{R}^{n}$
$f_{3}(x)=x x^{\top}$，$x\in \mathbb{R}^{n}$
    - （a）$\frac{\partial f_{i}}{\partial x}$的维度是什么？
        - （b）计算 Jacobi 矩阵。
        5.6对$f$关于$t$求导，对$g$关于$x$求导，其中$f(t)=\sin(\log(t^{\top}t))$，$t\in \mathbb{R}^{D}$，$g(X)=tr(AXB)$，$A\in \mathbb{R}^{D\times E}$，$X\in \mathbb{R}^{E\times F}$，$B\in \mathbb{R}^{F\times D}$，其中$tr(.)$表示迹。
        5.7使用链式法则计算以下函数的导数$df/dx$。提供每个单个偏导数的维度。详细描述你的步骤。
        - （a）$f(z)=\log(1 + z)$，$z = x^{\top}x$，$x\in \mathbb{R}^{D}$
        - （b）$f(z)=\sin(z)$，$z = Ax + b$，$A\in \mathbb{R}^{E\times D}$，$x\in \mathbb{R}^{D}$，$b\in \mathbb{R}^{E}$，其中$\sin(.)$应用于$z$的每个元素。
        5.8计算以下函数的导数$df/dx$。详细描述你的步骤。
        - （a）使用链式法则。提供每个单个偏导数的维度。不需要显式计算偏导数的乘积。$f(z)=\exp(-\frac{1}{2}z)$，$z = g(y)=y^{\top}S^{-1}y$，$y = h(x)=x - \mu$，其中$x,\mu\in \mathbb{R}^{D}$，$S\in \mathbb{R}^{D\times D}$。
        - （b）$f(x)=tr(xx^{\top}+\sigma^{2}I)$，$x\in \mathbb{R}^{D}$。这里$tr(A)$是$A$的迹，即对角元素$A_{ii}$的和。提示：显式写出外积。
        - （c）使用链式法则。提供每个单个偏导数的维度。不需要显式计算偏导数的乘积。$f = \tanh(z)\in \mathbb{R}^{M}$，$z = Ax + b$，$x\in \mathbb{R}^{N}$，$A\in \mathbb{R}^{M\times N}$，$b\in \mathbb{R}^{M}$。这里，$\tanh$应用于$z$的每个组件。
        5.9我们定义$g(z,\nu):=\log p(x,z)-\log q(z,\nu)$，$z := t(\epsilon,\nu)$，对于可微函数$p,q,t$以及$x\in \mathbb{R}^{D}$，$z\in \mathbb{R}^{E}$，$\nu\in \mathbb{R}^{F}$，$\epsilon\in \mathbb{R}^{G}$。使用链式法则计算梯度$\frac{d}{d\nu}g(z,\nu)$。