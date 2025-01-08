# Model Descriptions

This document contains descriptions[^ai_translate] of the two models referenced across this repository:

* The Demographic Fiscal Model[^dfm]
* The Demographic Wealth Model[^dwm]

[^dwm]: Wittmann, Lukas, and Christian Kuehn. "The Demographic-Wealth model for cliodynamics." Plos one 19, no. 4 (2024): e0298318.

[^dfm]: Turchin, Peter. "Historical dynamics: Why states rise and fall." (2018): 121-131.

Analyses of model dynamics provided by the authors of the models are not included.

[^ai_translate]: In order to reduce manual labor, the author had AI systems convert screenshots of PDF text descriptions to markdown; while the author has yet to find errors in the translation, there may be some. It is also work noting that _The Demographic Wealth Model_ paper had a moderate number of grammatical errors in its published form, so those may have been retained by the AI translator. The `typos` pre-commit hook did not capture any spelling errors within this file.

## The Demographic Fiscal Model

> I begin with the simplest possible model of agrarian state collapse. My purpose is first to expose the most basic logical blocks of the argument, and only then to start adding real-life complexities to it. In this section I will ask the reader to work through the model derivation and analysis with me, because the math is very simple, but at the same time crucial for understanding the main result. In the following sections, math will be banished to the Appendix.
>
> There are two variables in the model, one for the population and the other for the state. \(N(t)\) is the population density of subjects at time \(t\). I assume that the state area does not change, so the density is simply the population number divided by the area. Thus, the units of \(N\) could be individuals per \(km^2\). \(S(t)\) is the current accumulated state resources, which I measure in kilograms (or tons) of grain. The choice of this particular variable is based on the economic nature of the agrarian state, in which food is the main commodity produced.
>
> To start deriving the equations connecting the two structural variables (\(N\) and \(S\)), I first assume that the per capita rate of surplus production is a declining function of population numbers (this is David Ricardo’s law of diminishing returns). There are several socioecological mechanisms that underlie this relationship. First, as population grows, the stock of the most fertile land is exhausted, and increasingly more marginal plots are brought into cultivation. Second, increased population growth also means that land is subdivided into smaller parcels. Although each parcel receives more labor, its production rate (kg grain per ha per year) is also subject to the law of diminishing returns.
>
> For the purpose of simplicity, I will approximate the relationship between per capita rate of surplus production, \(\rho\), and population numbers, \(N\), with a linear function:
>
> $$
> \rho(N) = c_1 (1 - N / k)
> $$
>
> Here \(c_1\) is some proportionality constant, and \(k\) is the population size at which the surplus equals zero. Thus, for \(N > k\), the surplus is negative (the population produces less food than is needed to sustain it).
>
> Next, I assume that population dynamics are Malthusian:
>
> \[
> \dot{N} = rN
> \]
>
> and that the per capita rate of population increase is a linear function of the per capita rate of surplus production, \(r = c_2 \rho(N)\). Putting together these two assumptions, we arrive at a logistic model for population growth:
>
> \[
> \dot{N} = r_0 N \left(1 - \frac{N}{k}\right) \tag{7.1}
> \]
>
> where \(r_0 = c_1 c_2\) is the intrinsic rate of population growth, obtained when \(N\) is near 0. The parameter \(k\) is now seen as the "carrying capacity," or equilibrial population size. When \(N < k\), the society generates surplus, and the population grows. If \(N > k\), then the society produces less food than is needed for households to sustain and replace themselves, resulting in population decline.
>
> Turning now to the differential equation for state resources, \(S\), we note that \(S\) changes as a result of two opposite processes: revenues and expenditures. I will assume that the state collects a fixed proportion of surplus production as taxes. The total rate of surplus production is the product of per capita rate and population numbers. Thus, the taxation rate is \(c_3 \rho(N) N\), where \(c_3\) is the proportion of surplus collected as taxes. State expenditures are assumed to be proportional to the population size. The reason for this assumption is that, as population grows, the state must spend more resources on the army to protect and police it, on bureaucracy to collect and administer taxes, and on various public works (public buildings, roads, irrigation canals, etc). Putting together these processes we have the following equation for \(S\):
>
> \[
> \dot{S} = \rho_0 N \left(1 - \frac{N}{k}\right) - \beta N \tag{7.2}
> \]
>
> where \(\rho_0 = c_1 c_3\) is the per capita taxation rate at low population density and \(\beta\) the per capita state expenditure rate.
>
> We are not done yet. Although we have established the dynamic link from \(N\) to \(S\), there is no feedback effect (from \(S\) to \(N\)) in the model. I assume that the strong state has a positive effect on population dynamics; specifically, it increases \(k\), the sustainable population size given the ecological conditions and the current development of agricultural technology, as discussed in the previous section.
>
> Thus, the carrying capacity \(k\) is a monotonically increasing function of \(S\). However, \(k\) cannot increase without bound. No matter how high \(S\) is, at some point all potential land is brought into cultivation, and all potential improvements have been put in place. Thus, there must be some maximum \(k_{\text{max}}\), given the historically current level of agricultural technology. Another way of thinking about this mechanism is that the return on capital investment is also subject to a law of diminishing returns. I assume the following specific functional form for \(k(S)\):
>
> \[
> k(S) = k_0 \left(1 + c \frac{S}{s_0 + S}\right) \tag{7.3}
> \]
>
> The parameter \(k_0\) is the carrying capacity of the stateless society, \(c = k_{\text{max}} - k_0\) is the maximum possible gain in increasing \(k\) given unlimited funds, and \(s_0\) is a scaling constant.
>
> Putting together equations (7.1)–(7.3) we have the complete **demographic-fiscal model** (because it focuses on the fiscal health of the state as the main structural variable). The model has six parameters: \(\rho_0\), \(\beta\), \(r\), \(k_0\), \(c\), and \(s_0\), but we can reduce this set to four, by scaling \(N' = N / k_0\) and \(S' = S / \rho_0\). This procedure leaves us with the following parameters: \(\beta' = \beta / \rho_0\), \(r\), \(c\), and \(s' = s_0 / \rho_0\). The scaled equations are:
>
> \[
> \dot{N} = rN \left(1 - \frac{N}{k(S)}\right)
> \]
>
> \[
> \dot{S} = N \left(1 - \frac{N}{k(S)}\right) - \beta N \tag{7.4}
> \]
>
> \[
> k(S) = 1 + c \frac{S}{s_0 + S}
> \]
>
> (where I dropped the primes for better readability). I also impose the condition \(S \geq 0\) (that is, the state is not allowed to go into debt).

## The Demographic Wealth Model

> The main difference in the interpretation of the model compared to the DFM is the state’s role in it. In the DFM, the state’s surplus is determined through revenues and expenditures, measured in taxes, respectively more money that has to be spent in order to maintain the state’s infrastructure with growing population numbers.
>
> In our "Demographic-Wealth model" (DWM), the state’s surplus/wealth is measured in wealth gain and wealth loss, and the gains are mainly determined by two aspects:
>
> - **Taxes that are collected from the population.**
> - **Wealth that is generated with existing surplus**, for example, land gain through warfare, trade, or strategic investments.
>
> In addition, the wealth loss is mainly on the state’s wealth level. The more wealth the state has (more money or more land and therefore gaining more attention), the more expenditures it has to make in order to secure this wealth (from attacks, land loss, and maintaining infrastructure), similar to the theory of Olson.
>
> Starting with the dynamics for \(N\), as in the DFM, a logistic growth for the population is assumed, with \(r\) being the intrinsic rate of population growth. So, in the absence of the state, the population will grow until its “carrying capacity” \(k\).
>
> The carrying capacity \(k\) is a functional response to the state’s wealth,
>
> \[
> k(S) = k_0 + cS, \tag{2}
> \]
>
> meaning that more wealth and therefore more land and financial possibilities lead to more space and resources to live. The parameter \(k_0\) is the carrying capacity in the absence of the state, and \(c\) determines the dependence of the carrying capacity on the change in the state’s wealth. The difference to the functional \(k(S)\) in the DFM comes from the interpretation of \(S\) as the wealth of the state (land gain). A new part for the dynamics of \(N\) is the negative feedback effect from \(S\) to \(N\), inspired by Olson. A state that is growing in wealth has at some point a negative influence on population numbers. For example, one may consider the scenarios:
>
> - **Growing wealth leads to growing expenditures**, which lead to exploitation of the population.
> - **Growing wealth leads to more warfare**, and therefore to a higher death rate or emigration.
>
> In addition, with growing population and only a limited amount of food and living space available and taxes that have to be paid, a growing fraction of the population cannot afford living in the state. So, there will be a growing fraction of the population that leaves the state or dies. On the other hand, the remaining people have more resources and space, so it is assumed that the decreasing rate approaches one. Together with the negative feedback effect from \(S\) to \(N\), this behavior is described by the term \(-\alpha X(N, S)\). In this model, the functional \(X(N, S) = \frac{SN}{d + N}\) is chosen, because of the situation described above, where \(d > 0\) controls the strength of the negative feedback from \(S\) to \(N\) in the usual way of a Holling type-II response. Overall, the dynamics of \(N\) have the form:
>
> \[
> \dot{N} = rN \left(1 - \frac{N}{k(S)}\right) - \alpha S \frac{N}{d + N}. \tag{3}
> \]
>
> The dynamics of \(S\) are determined by wealth gain and wealth loss of two parties, the population and the state itself. The state collects taxes and can reinvest a portion of the surplus gained through the population for some extra wealth, for example, through loans, warfare, or land gain. This results in the term \(gSN\), with \(g = \tau \rho\), \(\tau\) being the tax rate, and \(\rho\) the fraction of the surplus that is gained through investing/expanding. But there are also expenditures that the state has to make. The larger the country and the more wealth the state has, the more money it has to spend for protection or maintaining the wealth. For example, if the state gains wealth through capturing new land, it has to pay additional attention to protect the new land by paying more soldiers and civil servants. In addition, it needs to provide a suitable living space, so it has to reinvest a growing amount of money into the infrastructure of the country. In summary, we consider that the dynamics of \(S\) has the form:
>
> \[
> \dot{S} = gSN - \beta S, \tag{4}
> \]
>
> with \(\beta\) being the fraction of the wealth that has to be spent. Together with the Eqs (2) and (3), the "Demographic-Wealth model" (DWM) is given by:
>
> \[
> \dot{N} = rN \left(1 - \frac{N}{k_0 + cS}\right) - \alpha S \frac{N}{d + N},
> \]
>
> \[
> \dot{S} = gSN - \beta S. \tag{5}
> \]
>
> Note that different choices for the interaction functions between population and state are definitely possible, e.g., the nonlinearities may also carry different powers for \(S\) and \(N\). Here we have effectively applied the principle to start with the simplest non-trivial case of a direct production interaction leading to the terms \(SN\).
