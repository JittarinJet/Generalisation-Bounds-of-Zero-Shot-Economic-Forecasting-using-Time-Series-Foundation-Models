# Generalisation-Bounds-of-Zero-Shot-Economic-Forecasting-using-Time-Series-Foundation-Models

This study investigates zero-shot forecasting capabilities of Time Series Foundation Models (TSFMs) for macroeconomic indicators. We apply TSFMs to forecasting economic indicators under univariate conditions, bypassing the need for train bespoke econometric models using and extensive training datasets. Our experiments were conducted on a case study dataset, without additional customisation. We rigorously back-tested three state-of-the-art TSFMs (Chronos, TimeGPT and Moirai) under data-scarce conditions and structural breaks. Our results demonstrate that appropriately engineered TSFMs can internalise rich economic dynamics, accommodate regime shifts, and deliver well-behaved uncertainty estimates out of the box, while matching state-of-the-art multivariate models on this domain. Our findings suggest that, without any fine-tuning, TSFMs can match or exceed classical models during stable economic conditions. However, they are vulnerable to degradation in performances during periods of rapid shocks. The findings offer guidance to practitioners on when zero-shot deployments are viable for macroeconomic monitoring and strategic planning.

**Citation**

```
@Article{make7040135,
AUTHOR = {Jetwiriyanon, Jittarin and Susnjak, Teo and Ranathunga, Surangika},
TITLE = {Generalisation Bounds of Zero-Shot Economic Forecasting Using Time Series Foundation Models},
JOURNAL = {Machine Learning and Knowledge Extraction},
VOLUME = {7},
YEAR = {2025},
NUMBER = {4},
ARTICLE-NUMBER = {135},
URL = {https://www.mdpi.com/2504-4990/7/4/135},
ISSN = {2504-4990},
ABSTRACT = {This study investigates the transfer learning capabilities of Time-Series Foundation Models (TSFMs) under the zero-shot setup, to forecast macroeconomic indicators. New TSFMs are continually emerging, offering significant potential to provide ready-trained and accurate forecasting models that generalise across a wide spectrum of domains. However, the transferability of their learning to many domains, especially economics, is not well understood. To that end, we study TSFM’s performance profile for economic forecasting, bypassing the need for training bespoke econometric models using extensive training datasets. Our experiments were conducted on a univariate case study dataset, in which we rigorously back-tested three state-of-the-art TSFMs (Chronos, TimeGPT, and Moirai) under data-scarce conditions and structural breaks. Our results demonstrate that appropriately engineered TSFMs can internalise rich economic dynamics, accommodate regime shifts, and deliver well-behaved uncertainty estimates out of the box, while matching and exceeding state-of-the-art multivariate models currently used in this domain. Our findings suggest that, without any fine-tuning and additional multivariate inputs, TSFMs can match or outperform classical models under both stable and volatile economic conditions. However, like all models, they are vulnerable to performance degradation during periods of rapid shocks, though they recover the forecasting accuracy faster than classical models. The findings offer guidance to practitioners on when zero-shot deployments are viable for macroeconomic monitoring and strategic planning.},
DOI = {10.3390/make7040135}
}
```

```
@misc{jetwiriyanon2025generalisationboundszeroshoteconomic,
      title={Generalisation Bounds of Zero-Shot Economic Forecasting using Time Series Foundation Models}, 
      author={Jittarin Jetwiriyanon and Teo Susnjak and Surangika Ranathunga},
      year={2025},
      eprint={2506.15705},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.15705}, 
}
```
