# 习题
## 1 
考虑一元函数
$$
f(x) = x^{3} + 6x^{2} - 3x - 5
$$
求它的稳定点，并分析它们是极小值、极大值还是鞍点

## 2 
考虑公式 (7.15) 中的 SGD 更新规则，写出批量大小为 $1$ 时的更新公式

## 3 
判断正误
* 任意两个凸集的交还是凸集
* 任意两个凸集的并还是凸集
* 两个凸集 $A$ 和 $B$，差集 $A - B$ 还是凸集

## 4 
判断正误
* 两个凸函数的和还是凸函数
* 两个凸函数的差还是凸函数
* 两个凸函数的乘积还是凸函数
* 两个凸函数 $f$ 和 $g$，则 $\max\{ f, g \}$ 还是凸函数

## 5 
将下面的优化问题转化为矩阵形式的线性优化问题
$$
\max\limits_{\boldsymbol{x} \in \mathbb{R}^{2},\; \xi \in \mathbb{R}}~\boldsymbol{p}^{\top}\boldsymbol{x} + \xi
$$
其中约束是 $\xi \geqslant 0, x_{0} \leqslant 0, x_{1} \leqslant 3$。

## 6
考虑图 7.9 中所示的线性规划问题：
$$
\begin{align}
\min\limits_{\boldsymbol{x} \in \mathbb{R}^{2}}~&~-\begin{bmatrix}
5\\3
\end{bmatrix}^{\top}\begin{bmatrix}
x_{1}\\x_{2}
\end{bmatrix}\\
\text{subject to}~&~ \begin{bmatrix}
2&2\\2&-4\\-2&1\\0&-1\\0&1
\end{bmatrix}\begin{bmatrix}
x_{1}\\x_{2}
\end{bmatrix} \leqslant \begin{bmatrix}
33\\8\\5\\-1\\8
\end{bmatrix}
\end{align}
$$
使用 Lagrangre 对偶求该问题的对偶线性规划问题

## 7 
考虑图 7.4 中所示的二次规划问题：
$$
\begin{align}
\min\limits_{\boldsymbol{x}\in \mathbb{R}^{2}}~&~ \frac{1}{2}\begin{bmatrix}
x_{1}\\x_{2}
\end{bmatrix}^{\top}\begin{bmatrix}
2&1\\1&4
\end{bmatrix}\begin{bmatrix}
x_{1}\\x_{2}
\end{bmatrix} + \begin{bmatrix}
5\\3
\end{bmatrix}^{\top}\begin{bmatrix}
x_{1}\\x_{2}
\end{bmatrix}\\
\text{subject to}~&~\begin{bmatrix}
1&0\\-1&0\\0&1\\0&-1
\end{bmatrix}\begin{bmatrix}
x_{1}\\x_{2}
\end{bmatrix} \leqslant \begin{bmatrix}
1\\1\\1\\1
\end{bmatrix}
\end{align}
$$
使用 Lagrangre 对偶求该问题的对偶二次规划问题

## 8
考虑下面的凸优化问题
$$
\begin{align}
\min\limits_{\boldsymbol{w} \in \mathbb{R}^{D}}~&~ \boldsymbol{w}^{\top}\boldsymbol{w}\\
\text{subject to}~&~ \boldsymbol{w}^{\top}\boldsymbol{x} \geqslant 1.
\end{align}
$$
引入 Lagrange 乘子 $\lambda$，求该问题的 Lagrangre 对偶

## 9
考虑向量 $\boldsymbol{x} \in \mathbb{R}^{D}$ 的负熵
$$
f(\boldsymbol{x}) = \sum\limits_{d=1}^{D} x_{d}\log x_{d}.
$$
假设我们使用的内积是标准内积，求它的凸共轭函数 $f^{*}(s)$

> **提示**
> 考虑某个函数，并令其梯度为零


## 10
考虑下面的函数
$$
f(\boldsymbol{x}) = \frac{1}{2}\boldsymbol{x}^{\top}\boldsymbol{A}\boldsymbol{x} + \boldsymbol{b}^{\top}\boldsymbol{x} + c,
$$
其中 $\boldsymbol{A}$ 是严格正定矩阵（可逆）。求 $f(\boldsymbol{x})$ 的凸共轭

## 11
常用于支持向量机（SVM）的铰链损失的形式如下：
$$
L(\alpha) = \max~\{ 0, 1-\alpha \},
$$
如果我们想要用梯度方法（例如 L-BGFS）求其最小值，并避免用到次梯度，我们需要对其不可微点 “光滑” 化。计算铰链损失的凸共轭 $L^{*}(\beta)$（其中 $\beta$ 是对偶变量），加上一个 $\ell_{2}$ 邻近项，然后再计算下面函数的凸共轭
$$
L^{*}(\beta) + \frac{\gamma}{2} \beta^{2},
$$
其中 $\gamma$ 是超参数。