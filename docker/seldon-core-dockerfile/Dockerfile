FROM wine-classifier-image:latest

EXPOSE 9000
ARG MODEL_NAME
## Define environment variable
ENV MODEL_NAME WineClassifierModel
ENV SERVICE_TYPE MODEL
ENV PERSISTENCE 0
CMD exec seldon-core-microservice $MODEL_NAME --service-type $SERVICE_TYPE --persistence $PERSISTENCE