# Protocolo de contraste emocional (Lia TTS)

O LoRA emocional só aprende emoção se a MESMA frase aparecer com emoções DIFERENTES.
Sem isso, o modelo decora o conteúdo e ignora o rótulo (é o que os testes mostraram).

## Como gravar

1. Crie a pasta `contrast_extra/` dentro de `/content/lia_qwen_lora/` no Colab
   (ou deixe os `.wav` na pasta do seu PC e suba para lá).
2. Um arquivo por frase, nomeando `{emoção}__{id}.wav`:
   - emoções aceitas: `alegre`, `triste`, `irritada`, `neutro`
   - ids: `01` a `08` (a frase correspondente abaixo)
   - exemplo: `alegre__01.wav`, `irritada__07.wav`
3. Fale TODAS as 8 frases em CADA uma das 4 emoções = 32 arquivos.
   Mínimo para valer a pena: frases `01`–`04` nas 4 emoções = 16 arquivos.
4. Gravação: ambiente silencioso, boca perto do microfone, volume constante,
   uma frase por arquivo, sem eco. Qualquer formato serve — o pipeline converte
   para mono 24 kHz sozinho. Máximo 12 s por arquivo.

## As frases (grave exatamente assim, sem mudar palavras)

| id | Frase |
|----|-------|
| 01 | Tudo certo por aqui, pode continuar. |
| 02 | Vamos ver o que temos para hoje. |
| 03 | Aqui está o resumo que você pediu. |
| 04 | Eu já deixei tudo pronto na mesa. |
| 05 | O arquivo está na pasta que você abriu antes. |
| 06 | O resultado final aparece na próxima tela. |
| 07 | A reunião começa dentro de alguns minutos. |
| 08 | Me conta como foi o seu dia. |

Não envie nada a serviço externo: os arquivos vão só do seu computador para o
seu runtime do Colab.

## O que acontece depois

Rode o ONE-CLICK RUNNER: ele ingere a pasta, re-treina (os dados mudaram, o
cache invalida sozinho) e termina na escada de épocas. Ouça a escada, diga a
época vencedora, e o A/B completo de 3 frases roda em seguida.
