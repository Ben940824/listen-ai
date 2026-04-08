FROM golang:1.24

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN go build -o stat-service .

EXPOSE 8002

CMD ["./stat-service"]
