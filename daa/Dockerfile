FROM alpine:latest

RUN apk update && \
    apk add --no-cache build-base cmake make && \
    rm -rf /var/cache/apk/*

WORKDIR /app

COPY ./app/CMakeLists.txt ./app/main.cpp ./
COPY ./app/src/ ./src/
COPY ./app/include/ ./include/

RUN mkdir build && \
    cd build && \
    cmake .. && \
    make && \
    cp web_server .. && \
    cd .. && \
    rm -rf build

EXPOSE 8080

CMD ["./web_server"]