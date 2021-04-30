FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
RUN python -m pip install numba

COPY . .

RUN for i in "compute" "cholesky2" "go_fast"; do python run_benchmark.py -b $i -f numpy; done
RUN for i in "compute" "cholesky2" "go_fast"; do python run_benchmark.py -b $i -f numba; done
RUN python plot_lines.py
RUN python plot_results.py
