#!/bin/sh

mkdir -p /etc/kafka && cp /kafka_server_jaas.conf /etc/kafka/ && sed -i "s/\${KAFKA_PASSWORD}/${KAFKA_PASSWORD}/g" /etc/kafka/kafka_server_jaas.conf && sed -i "s/\${KAFKA_USERNAME}/${KAFKA_USERNAME}/g" /etc/kafka/kafka_server_jaas.conf && exec start-kafka.sh