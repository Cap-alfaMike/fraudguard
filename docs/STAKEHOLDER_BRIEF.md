# FraudGuard: resumo para a diretoria

## O problema em uma frase

Cada fraude que passa custa o valor roubado, a disputa e, muitas vezes, o cliente; cada cliente honesto bloqueado custa atrito e, às vezes, o mesmo cliente.

## O que o FraudGuard faz

Avalia cada transação em menos de um milésimo de segundo e decide se ela segue ou vai para revisão. Ao lado disso, um sistema de alerta antecipado observa o fluxo o tempo todo e avisa quando algo foge do normal: um ataque em andamento, uma mudança de comportamento, uma lentidão. Quando isso acontece, gera um relatório em linguagem de negócio, com números verificáveis e as ações recomendadas.

## O que os números mostram

Em um período de teste que o modelo nunca tinha visto:

- **3 de cada 4 fraudes foram barradas**, e elas representavam **87% do dinheiro em risco**. O modelo foi desenhado para pesar o valor: prioriza as fraudes caras.
- **Apenas 1 em cada 900 clientes honestos** foi encaminhado para revisão.
- **O custo total da fraude caiu 76,5%**, contando perdas, disputas, custo de revisão e clientes perdidos.
- **€134 economizados a cada 1.000 transações.**

A 3.000 transações por minuto, a projeção anual vai de **€44 milhões a €212 milhões**, conforme a taxa de fraude real da nossa base seja um quarto ou igual à do dataset de referência. O número exato depende de dois fatores que precisamos validar juntos: a taxa de fraude da nossa carteira e as premissas de custo (quanto vale um cliente, quantos saem depois de sofrer uma fraude).

## Por que isso protege a marca

Um banco que deixa fraudes passar perde a coisa mais difícil de reconstruir: a sensação de que o dinheiro está seguro ali. O FraudGuard trata esse risco como custo explícito. Pelas premissas atuais, a cada 1 cliente que sai incomodado por um bloqueio indevido, mais de 5 que sairiam após sofrer fraude são preservados.

## Por que confiar

- **Cada decisão é explicável:** o sistema diz quais sinais pesaram em cada transação.
- **A inteligência artificial generativa não decide nada:** ela só escreve os relatórios, a partir de números agregados, sem dados de clientes, e o sistema verifica se ela não inventou valores. Se ela ficar indisponível, um relatório padrão é gerado no lugar.
- **As premissas estão à vista:** os custos usados nas decisões são parâmetros que Finanças e Risco podem ajustar em minutos, sem refazer o modelo.
- **Foi testado como produção:** 122 testes automáticos, carga real medida, container com as práticas de segurança do mercado.

## O que é preciso para ir a produção

1. Rodar o mesmo pipeline sobre os dados reais (o sistema já está pronto para isso) e confirmar os números.
2. Validar as premissas de custo com Finanças e Risco.
3. Definir com Operações a capacidade da fila de revisão e o protocolo para alertas críticos.
4. Operar em modo sombra (pontua sem decidir) por uma semana antes de ligar.
