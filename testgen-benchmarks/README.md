# testgen-benchmarks
Benchmarking tools for Automated Test Automation Generation ATAG: [https://github.com/rikulehtonen/ATAG](https://github.com/rikulehtonen/ATAG)

![Demo](material/atag_demo.gif)

## Installation

1. Follow instructions in [https://github.com/rikulehtonen/ATAG/tree/main#installation](https://github.com/rikulehtonen/ATAG/tree/main#installation)

2. Install benchmark environment specific tools

## Usage

DETAILED INSTRUCTIONS ARE UPDATED LATER!


### Login Benchmark

Run benchmark:

```bash
cd /login-demo
python train.py
```

### Shopping Cart Benchmark

Install node.js: [https://nodejs.org/en/download](https://nodejs.org/en/download)

Build and start the web app

```bash
cd cart-demo/demostore/
npm run build
npm start
```

Run benchmark while app is running:

```bash
cd cart-demo
python train.py
```