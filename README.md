[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
<!-- 
***[![MIT License][license-shield]][license-url]
-->

<!-- PROJECT LOGO -->

<br />
<div align="center">
  <a href="https://github.com/microsoft/robustlearn">
    <img src="https://wjdcloud.blob.core.windows.net/tools/roblearn.png" alt="Logo" width="400">
  </a>

  <strong>robustlearn</strong>: A unified library for research on robust machine learning

</div>

Latest research in robust machine learning, including adversarial/backdoor attack and defense, out-of-distribution (OOD) generalization, and safe transfer learning.

Hosted projects:
- **Diversify** (ICLR 2023, #OOD):
  - [Code](./diversify/) | [Out-of-distribution Representation Learning for Time Series Classification](https://arxiv.org/abs/2209.07027)
- **DRM** (KDD 2023, #OOD):
  - [Code](./drm/) | [Domain-Specific Risk Minimization for Out-of-Distribution Generalization](https://arxiv.org/abs/2208.08661)
- **SDMix** (IMWUT 2022, #OOD): 
  - [Code](./sdmix/) | [Semantic-Discriminative Mixup for Generalizable Sensor-based Cross-domain Activity Recognition](http://arxiv.org/abs/2206.06629)
- **MARC** (ACML 2022, #Long-tail): 
  - [Code](./marc/) | [Margin Calibration for Long-Tailed Visual Recognition](https://arxiv.org/abs/2112.07225)
- **FedCLIP** (IEEE Data Engineering Bulletin 2023, #OOD #LargeModel): 
  - [Code](./fedclip/) | [FedCLIP: Fast Generalization and Personalization for CLIP in Federated Learning](https://arxiv.org/abs/2302.13485)
- **ChatGPT robustness** (arXiv 2023, #OOD #Adversarial #LargeModel): 
  - [Code](./chatgpt-robust/) | [On the Robustness of ChatGPT: An Adversarial and Out-of-distribution Perspective](https://arxiv.org/abs/2302.12095)
- Stay tuned for more upcoming projects!

You can clone or download this repo. Then, go to the project folder that you are interested to run and develop your research.

Related repos:
  - Transfer learning: [[transferlearning: everything for transfer, domain adaptation, and more](https://github.com/jindongwang/transferlearning)]
  - Semi-supervised learning: [[USB: unified semi-supervised learning benchmark](https://github.com/microsoft/Semi-supervised-learning)] | [[TorchSSL: a unified SSL library](https://github.com/TorchSSL/TorchSSL)] 
  - Prompt benchmark for large language models: [[PromptBench: adverarial robustness of prompts of LLMs](https://github.com/microsoft/promptbench)]
  - Evlauation of large language models: [[LLM-eval](https://llm-eval.github.io/)]
  - Federated learning: [[PersonalizedFL: library for personalized federated learning](https://github.com/microsoft/PersonalizedFL)]


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.


[contributors-shield]: https://img.shields.io/github/contributors/microsoft/robustlearn.svg?style=for-the-badge
[contributors-url]: https://github.com/microsoft/robustlearn/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/microsoft/robustlearn.svg?style=for-the-badge
[forks-url]: https://github.com/microsoft/robustlearn/network/members
[stars-shield]: https://img.shields.io/github/stars/microsoft/robustlearn.svg?style=for-the-badge
[stars-url]: https://github.com/microsoft/robustlearn/stargazers
[issues-shield]: https://img.shields.io/github/issues/microsoft/robustlearn.svg?style=for-the-badge
[issues-url]: https://github.com/microsoft/robustlearn/issues
[license-shield]: https://img.shields.io/github/license/microsoft/robustlearn.svg?style=for-the-badge
[license-url]: https://github.com/microsoft/robustlearn/blob/main/LICENSE.txt