## 练习

6.1 考虑以下两个离散随机变量$X$和$Y$的双变量分布$p(x,y)$。

计算：

a. 边缘分布$p(x)$和$p(y)$。

b. 条件分布$p(x|Y=y_1)$和$p(y|X=x_3)$。

6.2 考虑两个高斯分布的混合（如图6.4所示），

$$0.4\mathcal{N}\left(\begin{bmatrix}10\\2\end{bmatrix},\begin{bmatrix}1&0\\0&1\end{bmatrix}\right)+0.6\mathcal{N}\left(\begin{bmatrix}0\\0\end{bmatrix},\begin{bmatrix}8.4&2.0\\2.0&1.7\end{bmatrix}\right).$$

a. 计算每个维度的边缘分布。

b. 计算每个边缘分布的均值、众数和中位数。

c. 计算二维分布的均值和众数。

6.3 你编写了一个计算机程序，该程序有时会编译成功，有时会编译失败（代码没有改变）。你决定使用参数为$\mu$的伯努利分布来模拟编译器的这种随机性（成功与不成功）$x$：

$$p(x|\mu)=\mu^x(1-\mu)^{1-x},\quad x\in\{0,1\}.$$

为伯努利似然选择一个共轭先验，并计算后验分布$p(\dot{\mu}|x_1,\ldots,x_N)$。

6.4 有两个袋子。第一个袋子里有四个芒果和两个苹果；第二个袋子里有四个芒果和四个苹果。

我们还有一个有偏的硬币，正面朝上的概率为0.6，反面朝上的概率为0.4。如果硬币正面朝上，我们从袋子1中随机挑选一个水果；否则，从袋子2中随机挑选一个水果。

你的朋友抛了硬币（你看不到结果），从对应的袋子中随机挑选了一个水果，并给了你一个芒果。

从袋子2中挑选出这个芒果的概率是多少？

$提示：使用\textit{贝叶斯定理。}$

6.5 考虑时间序列模型

$$x_{t+1}=Ax_{t}+w,\quad w\sim\mathcal{N}(0,Q)\\y_{t}=Cx_{t}+v,\quad v\sim\mathcal{N}(0,R),$$

其中$w,v$是独立同分布的高斯噪声变量。进一步假设$p(x_0)=\mathcal{N}(\mu_{0},\Sigma_{0})$。

a. $p(x_0,x_1,\ldots,x_T)$的形式是什么？证明你的答案（你不需要显式地计算联合分布）。

b. 假设$p(x_t|y_1,\ldots,y_t)=\mathcal{N}(\mu_t,\Sigma_t)$。

1. 计算$p(x_{t+1}|y_1,\ldots,y_t)$。
2. 计算$p(x_{t+1},y_{t+1}|y_{1},\ldots,y_{t})$。
3. 在时间$t+1$，我们观察到值$y_{t+1}=\hat{y}$。计算条件分布$p(x_{t+1}|y_1,\ldots,y_{t+1})$。

6.6 证明（6.44）中的关系，该关系将方差的标准定义与方差的原始分数表达式联系起来。

6.7 证明（6.45）中的关系，该关系将数据集中示例之间的成对差异与方差的原始分数表达式联系起来。

6.8 将伯努利分布表示为指数族分布的自然参数形式，参见（6.107）。

6.9 将二项式分布表示为指数族分布。同样，将贝塔分布表示为指数族分布。证明贝塔分布和二项式分布的乘积也是指数族分布的成员。

**6.10 以两种方式推导出第6.5.2节中的关系**：

a. 通过完成平方

b. 通过将高斯分布表达为其指数族形式

两个高斯分布 $\mathcal{N}(x\mid a,\boldsymbol{A})\mathcal{N}(x\mid\boldsymbol{b},\boldsymbol{B})$ 的乘积是一个未归一化的高斯分布 $c\mathcal{N}(x\mid c,C)$，其中

$$\begin{aligned}

&C=(A^{-1}+B^{-1})^{-1}\\

&c=C(A^{-1}a+B^{-1}b)\\

&c=(2\pi)^{-\frac{D}{2}}\mid A+B\mid^{-\frac{1}{2}}\exp\left(-\frac{1}{2}(a-b)^{\top}(A+B)^{-1}(a-b)\right).

\end{aligned}$$

注意，归一化常数 $c$ 本身可以视为在 $a$ 或 $b$ 上的（归一化）高斯分布，具有“膨胀”的协方差矩阵 $A+B$，即 $c= \mathcal{N} ( a\mid b, A+ \boldsymbol{B}) = \mathcal{N} ( b\mid a, A+ B)$。

**6.11 迭代期望**

考虑两个具有联合分布 $p(x,y)$ 的随机变量 $x,y$。证明

$$\mathrm{E}_X[x]=\mathrm{E}_Y\left[\mathrm{E}_X[x\:|\:y]\right].$$

这里，$\mathbb{E}_X[x\mid y]$ 表示在条件分布 $p(x\mid y)$ 下 $x$ 的期望值。

**6.12 高斯随机变量的操作**

考虑一个高斯随机变量 $x\sim\mathcal{N}(x\mid\mu_x,\Sigma_x)$，其中 $x\in\mathbb{R}^D$。

此外，我们有

$$y=Ax+b+w\:,$$

其中 $y\in\mathbb{R}^E,A\in\mathbb{R}^{E\times D},b\in\mathbb{R}^E$，且 $w\sim\mathcal{N}(w\mid0,Q)$ 是独立的高斯噪声。“独立”意味着 $x$ 和 $w$ 是独立的随机变量，且 $Q$ 是对角的。

a. 写下似然 $p(\boldsymbol{y}\mid\boldsymbol{x})$。

b. 分布 $p(\boldsymbol{y})=\int p(\boldsymbol{y}\mid\boldsymbol{x})p(\boldsymbol{x})d\boldsymbol{x}$ 是高斯的。计算均值 $\mu_y$ 和协方差 $\Sigma_y$。详细推导你的结果。

c. 随机变量 $y$ 根据测量映射进行变换

$$z=Cy+v\:,$$

其中 $z\in\mathbb{R}^F,C\in\mathbb{R}^{F\times E}$，且 $v\sim\mathcal{N}(v\mid\mathbf{0},\boldsymbol{R})$ 是独立的高斯（测量）噪声。

\- 写下 $p(\boldsymbol z\mid\boldsymbol y)$。

\- 计算 $p(z)$，即均值 $\mu_z$ 和协方差 $\Sigma_z$。详细推导你的结果。

d. 现在，测量了一个值 $\hat{y}$。计算后验分布 $p(\boldsymbol x\mid\hat{\boldsymbol{y}})$。

**解题提示**：这个后验也是高斯的，即我们只需要确定其均值和协方差矩阵。首先明确计算联合高斯 $p(\boldsymbol x,\boldsymbol y)$。这也需要我们计算交叉协方差 Cov$*_{x,y}[**x,y**]$* *和 Cov$_{y,x}[y,x]**$**。然后应用高斯条件规则。*

**6.13 概率积分变换**

给定一个连续随机变量 $X$，其累积分布函数为 $F_X(x)$，证明随机变量 $Y:=F_X(X)$ 是均匀分布的（定理6.15）。