# Distância Euclidiana entre Vetores (Otimizada) - Disciplina de PAD

Trabalho desenvolvido para a disciplina de Processamento de Alto Desempenho (PAD) com o objetivo de calcular a distância euclidiana entre um vetor de referência `q` e um conjunto de vetores `X` com `N` vetores de dimensão `D`.

O valor de `D` é informado pelo usuário, enquanto `N` foi _hard-coded_ para 3,584 (assim como no [original](https://github.com/Equiel-1703/pad_distancia_euclidiana/)).

Esta versão foi implementada inteiramente em **C++** com a API **OpenCL**, configurada para executar em paralelo na **CPU** (`CL_DEVICE_TYPE_CPU`). Foi utilizado **Shared Virtual Memory (SVM)** para otimizar o compartilhamento e acesso à memória.

---

## Requisitos

* Compilador C++ com suporte a C++17 ou superior (ex: `g++` ou `clang++`).
* CMake 3.15 ou superior.
* Drivers e SDK OpenCL instalados com suporte a SVM (*Shared Virtual Memory*).

---

## Como Compilar e Executar

1. **Clone o repositório e acesse a pasta do projeto:**

   ```bash
   git clone https://github.com/Equiel-1703/pad_distancia_euclidiana_cpp.git
   cd pad_distancia_euclidiana_cpp
   ```
2. **Compile o código-fonte com o CMake:**
    ```bash
    cmake -S . -B build
    cmake --build build
    ```
3. **Execute o programa informando a dimensão D dos vetores:**
   ```bash
   ./build/distancia_euclidiana 32
   ```
4. **Para executar a versão com otimização matemática, use a flag `-f|--fast`:**
   ```bash
   ./build/distancia_euclidiana 32 -f
   ```

## Relatório

O relatório descrevendo a metodologia utilizada, os resultados obtidos com análise estatística e conclusões pode ser encontrado no arquivo `PAD_Relatório_2.pdf` na raiz do projeto.

## Licensa

Esse projeto está licenciado sob a Licença GNU General Public License v3.0.
