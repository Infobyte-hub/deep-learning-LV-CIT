# Avaliação de Classificadores de Imagem Multi-rótulo

---

## 📝 Agradecimentos e Créditos

Este projeto é uma **reprodução e extensão experimental** baseada no trabalho original:

* A Combinatorial Interaction Testing Method for Multi-Label Image Classifier
* Peng Wang, Shengyou Hu, Huayao Wu*, Xintao Niu, Changhai Nie, and Lin Chen
* 2024 IEEE 35th International Symposium on Software Reliability Engineering (ISSRE)
* 2024
* **Link para o Trabalho Original:** (https://github.com/GIST-NJU/LV-CIT)

Este estudo visa replicar e estender o experimento original de Peng Wang et al., apresentando novos datasets e arquiteturas de redes neurais profundas para validar a acurácia e a generalização do método LV-CIT em diferentes situações.

---

## 📂 Sobre Este Repositório

Este projeto é uma tentativa de reprodução e implementação dos experimentos originais de Peng Wang et al.,como parte de um estudo acadêmico de mestrado, que visa analisar o funcionamento de uma rede neural profunda.
Inicialmente a ideia era reproduzir e estender os experimentos originais.
Visto que os próprios autores notaram dificuldades por parte da comunidade científica na reprodução de seus experimentos, os próprios autores afirmaram que pode haver a presença de discrepâncias quanto a instalação de certas dependências e versões de software que podem gerar erros de execução. Deixando claro uma manobra para contorná-los. Nesta versão inicial do repositório, não foi possível a execução dos experimentos com sucesso dados aos problemas mensionados. Novas atualizações com o progresso da pesquisa e a resolução dos desafios serão adicionadas em versões futuras.

Este repositório contém os arquivos necessários para a implementação em Python para a execução do LV-CIT em testes black-box.

Este repositório (ultimas atualizações) contém os scripts, notebooks e arquivos auxiliares usados como parte de um estudo acadêmico de mestrado, cujo objetivo é analisar e testar o funcionamento de um pipeline de Deep Learning para testes combinatórios em classificadores multi-rótulo.

---

### ⚙️ O que já foi reproduzido

✅ Ambiente configurado no Google Colab, com Google Drive para armazenamento persistente.  
✅ Scripts ajustados (`compositer.py`) executados como módulo, gerando saídas estruturadas.  
✅ Execução de geração de imagens compostas com covering arrays (VOC e COCO).  
✅ Treino de uma ResNet18 simples para classificação multi-rótulo, com checkpoints automáticos para retomada.  
✅ Monitoramento via TensorBoard (parcial, pois nem sempre gera saídas visuais se o treino interrompe cedo).  
✅ Organização em blocos numerados no notebook principal, documentando cada etapa.  
✅ Anotações de limitações práticas encontradas (tempo, falta de GPU, falhas de compatibilidade).  
✅ Base para extensões futuras, integração com outros datasets e novas arquiteturas.

---

### 🚦 Status do Projeto
Atenção: Este repositório representa uma versão inicial e em desenvolvimento do meu trabalho de mestrado. Os experimentos descritos ainda não foram totalmente reproduzidos com sucesso devido a desafios de compatibilidade e configuração das dependências originais. Estou trabalhando ativamente na resolução desses problemas e na implementação das minhas extensões.

Atualizações futuras irão incluir:  
- Resultados mais robustos para o pipeline completo.  
- Scripts revisados para execução **end-to-end** em ambiente controlado.  
- Documentação passo a passo ainda mais detalhada.

### 🚀 Como Rodar o Projeto
Observação: Como mencionado na seção "Status do Projeto", a execução dos experimentos ainda apresenta desafios devido a dependências e compatibilidade de software. Estou trabalhando para documentar os passos de instalação e execução de forma mais robusta em futuras atualizações.
Para acessar os arquivos e o código-fonte desta versão, clone este repositório:`git clone https://github.com/Infobyte-hub/deep-learning-LV-CIT`

Para referência sobre o ambiente original e detalhes adicionais, consulte o trabalho original em: `git clone https://github.com/GIST-NJU/LV-CIT`


---

### Contato

Igor Goulart Cabral

Aluno de Mestrado em Informática
