---
layout: post
title: 数值计算方法第二版习题2第15题
categories: NMM
description: Numerical malculation method
keywords: Newton Iteration
---
本文是数值计算方法课程的一道简单的证明习题。它表明，在目标函数满足一定条件下，存在一种改进的牛顿法使得在用于求解方程或优化问题时至少是二阶收敛的。

#### 题目：

设函数$f(x)$在$[a,b]$上至少三阶连续可微，$p\in(a,b)$为$f(x)$的一个$m$重零点，求一个$\lambda$值使改进的$Newton$下山法  

$$
x_{k+1}=x_{k}-\lambda\frac{f(x_k)}{f'(x_k)}
$$

至少是二阶收敛的。

#### 求解：

首先，对于一个迭代过程$x_{k+1}=\varphi{(x_{k})}$，如果$\varphi^{(p)}(x)$在所求根$x^\*$的附近连续，并且 

$$
\varphi'(x^*)=\varphi''(x^*)=\cdots=\varphi^{(p-1)}(x^*)=0,\quad{\varphi^{(p)}(x^*)\ne0}
$$  

则该迭代过程在点$x^\*$附近是$p$阶收敛的。

此定理的证明可由$\varphi(x)$在根$x^*$附近做$Taylor$展开来证明，此处不详述。

因此，要证明一个迭代过程至少是二阶收敛的，我们只需证明$\varphi{(x)}$在其根$x^*$的一阶导数为$0$即可。

对于此题，我们首先将迭代过程标准化，即表示为$x_{k+1}=\varphi(x)$的形式。因此我们得到:
$$
\varphi(x)=x-\lambda\frac{f(x)}{f'(x)}\tag{1}
$$
而$p$为$f(x)=0$在$(a,b)$的一个根，因此若要证明改进的$Newton$下山法至少是二阶收敛的，只需要证明$\varphi'(p)=0$即可。

由$p$是$f(x)$的$m$重零点，因此我们可设$f(x)=(x-p)^{m}g(x)$，其中$g(x)$满足$g(p)\ne0$。

对$\varphi(x)$求一阶导有：

$$
\begin{aligned}
\varphi'(x)&=1-\lambda(\frac{[f'(x)]^2-f(x)f''(x)}{[f'(x)]^2})\\
&=1-\lambda+\lambda\frac{f(x)f''(x)}{[f'(x)]^2}
\end{aligned}
\tag{2}
$$

由于$f^{(j)}(p)=0$，其中$j\in[0,m-1]$。因此，$\varphi'(p)$中最后一项分子分母均为$0$，故必须表示为如下极限形式：

$$
\varphi'(p)=1-\lambda+\lambda\lim_{x\to{p}}\frac{f(x)f''(x)}{[f'(x)]^2}\tag{3}
$$

对$f(x)$分别求一阶和二阶导有：

$$
\begin{aligned}
f'(x)&=m(x-p)^{m-1}g(x)+(x-p)^{m}g'(x)\\
&=(x-p)^{m-1}[mg(x)+(x-p)g'(x)]\\
\end{aligned}\\
\begin{aligned}
f''(x)&=m[(m-1)(x-p)^{m-2}g(x)+(x-p)^{m-1}g'(x)]+m(x-p)^{m-1}g'(x)+(x-p)^{m}g''(x)\\
&=m(m-1)(x-p)^{m-2}g(x)+2m(x-p)^{m-1}g'(x)+(x-p)^{m}g''(x)\\
&=(x-p)^{m-2}[m(m-1)g(x)+2m(x-p)g'(x)+(x-p)^2g''(x)]
\end{aligned}
\tag{4}
$$

将式(4)代入式(3)，并将分子分母约分和代入，有：

$$
\begin{aligned}
\varphi'(p)&=1-\lambda+\lambda\lim_{x\to{p}}\frac{g(x)[m(m-1)g(x)+2m(x-p)g'(x)+(x-p)^2g''(x)]}{[mg(x)+(x-p)g'(x)]^2}\\
&=1-\lambda+\lambda\lim_{x\to{p}}\frac{m(m-1)g^2(x)}{m^2{g^2(x)}}\\
&=1-\lambda+\lambda\frac{m-1}{m}\\
&=1-\frac{\lambda}{m}
\end{aligned}
\tag{5}
$$

我们令$\varphi'(p)=0$，即可得到$\lambda=m$。

因此，当取$\lambda=m$时，改进的$Newton$下山法至少是二阶收敛的。

