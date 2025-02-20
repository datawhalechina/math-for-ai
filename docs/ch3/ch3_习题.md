## 习题

### 3.1
证明对所有的 $x = [x_1, x_2]^T \in \mathbb{R}^2$ 和 $y = [y_1, y_2]^T \in \mathbb{R}^2$ ，如下定义的函数  $\langle \cdot, \cdot \rangle$  是一个内积。
$$ \langle x, y \rangle := x_1y_1 - (x_1y_2 + x_2y_1) + 2(x_2y_2) $$

### 3.2
考虑带有如下定义之函数 $\langle \cdot, \cdot \rangle$ 的 $\mathbb{R}^2$ ，此函数是一个内积吗？
$$ \langle x, y \rangle := x^T \underbrace{\begin{bmatrix} 2 & 0 \\ 1 & 2 \end{bmatrix}}_{=:\boldsymbol{A}} y $$

### 3.3
用下列不同的内积定义计算 $x$ 和 $y$ 的距离：
$$ x = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix},\quad y = \begin{bmatrix} -1 \\ -1 \\ 0 \end{bmatrix} $$
a. $\langle x, y \rangle := x^Ty$
b. $\langle x, y \rangle := x^T A y$, $A := \begin{bmatrix} 2 & 1 & 0 \\ 1 & 3 & -1 \\ 0 & -1 & 2 \end{bmatrix}$


### 3.4
用下列不同的内积定义计算 $x$ 和 $y$ 的夹角：
$$ x = \begin{bmatrix} 1 \\ 2 \end{bmatrix},\quad y = \begin{bmatrix} -1 \\ -1 \end{bmatrix} $$
a. $\langle x, y \rangle := x^Ty$
b. $\langle x, y \rangle := x^T B y$, $B := \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}$

### 3.5
考虑装配点积的 Euclid 空间 $\mathbb{R}^5$ 。一个子空间 $U \subseteq \mathbb{R}^5$ 和一个向量 $x \in \mathbb{R}^5$ 如下：
$$ U = \text{span}\left[
\begin{bmatrix}
0 \\ -1 \\ 2 \\ 0 \\ 2
\end{bmatrix},
\begin{bmatrix}
1 \\ -3 \\ 1 \\ -1 \\ 2
\end{bmatrix},
\begin{bmatrix}
-3 \\ 4 \\ 1 \\ 2 \\ 1
\end{bmatrix},
\begin{bmatrix}
-1 \\ -3 \\ 5 \\ 0 \\ 7
\end{bmatrix}
\right], \quad x =
\begin{bmatrix}
-1 \\ -9 \\ -1 \\ 4 \\ 1
\end{bmatrix}
$$
a. 计算 $x$ 到 $U$ 的正交投影 $\pi_U(x)$ 
b. 计算 $x$ 到 $U$ 的距离 $d(x, U)$

### 3.6
考虑装配有如下内积的 $\mathbb{R}^3$ ：
$$ \langle x, y \rangle := x^T \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & -1 \\ 0 & -1 & 2 \end{bmatrix} y $$
记 $e_1, e_2, e_3$ 为 $\mathbb{R}^{3}$ 中的标准基。

a. 计算 $\boldsymbol{e}_{2}$ 至子空间 $U = \text{span}\{ \boldsymbol{e}_{1}, \boldsymbol{e}_{3} \}$ 的投影 $\pi_U(e_2)$ 
b. 计算 $\boldsymbol{e}_{2}$ 到 $U$ 的距离 $d(\boldsymbol{e}_2, U)$
c. 请绘制所有的标准正交基和 $d(\boldsymbol{e}_2, U)$

> 提示：正交性是由内积决定的

### 3.7
令 $V$ 为一向量空间， $\pi$ 是其上的一个自同态。
a. 证明：$\pi$ 是投影变换，当且仅当 $\text{id}_V - \pi$ 是一个投影变换，其中 $\text{id}_V$ 是 $V$ 上的单位同态。
b. 现假设 $\pi$ 是投影变换，计算 $\text{Im}(\text{id}_V - \pi)$ 和 $\text{ker}(\text{id}_V - \pi)$ 作为 $\text{Im}(\pi)$ 和 $\text{ker}(\pi)$ 的函数。

### 3.8
使用 Gram-Schmidt 正交化方法，将某二维子空间 $U \subseteq \mathbb{R}^3$ 的基 $B = (b_1, b_2)$ 转换为 $U$ 中的标准正交基 $C = (c_1, c_2)$，其中
$$ b_1 := \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix},\quad b_2 := \begin{bmatrix} -1 \\ 2 \\ 2 \end{bmatrix} $$

### 3.9
令 $n \in \mathbb{N}$ 同时令 $x_1, \ldots, x_n > 0$ 为 $n$ 个正实数，且满足 $x_1 + \ldots + x_n = 1$. 使用 Cauchy-Schwartz 不等式证明：
a. $\displaystyle \sum_{i=1}^{n} x_i^2 \geq 1$
b. $\displaystyle \sum_{i=1}^{n} \frac{1}{x_i} \geq n^2$

> 提示: 考虑 $\mathbb{R}^n$ 上的内积。然后选择恰当的 $x, y \in \mathbb{R}^n$。

### 3.10
将下列向量旋转 $30^{\circ}$。
$$ x_1 := \begin{bmatrix} 2 \\ 3 \end{bmatrix},\quad x_2 := \begin{bmatrix} 0 \\ -1 \end{bmatrix} $$
