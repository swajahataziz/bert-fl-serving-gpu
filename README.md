# BERT Flask Serving Container (GPU)

This repo contains the code for setting up a serving docker container using Flask and Waitress.

## Preparing the Docker image.

* Pre-requisites
	* An IAM user with Git credentials set up
	* Git client
	* Docker

* Clone the repo:
	* `git clone https://git-codecommit.us-east-2.amazonaws.com/v1/repos/who-bert-fl-serving`
	* `cd who-bert-fl-serving`

* Build the image:
	* `image=<your-image-name`
	* `docker build -t ${image}`

## Run the Image

* `nvidia-docker run -it -p 8000:8000 ${image}`

## Example of calling the inference APIs

* `payload=payload.json`
* `content=application/json`
* Online:
	* `curl --data-binary @${payload} -H "Content-Type: ${content}" -v http://localhost:8000/predict`
* Batch/Offline:
	* `curl --data-binary @${batch} -H "Content-Type: ${content}" -v http://localhost:8000/invocations`