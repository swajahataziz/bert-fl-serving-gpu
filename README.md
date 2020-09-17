# BERT Serving Container Using Flask & Waitress (GPU)

This repo contains the code for setting up a serving docker container using Flask and Waitress on a GPU instance. The model weights are available only in the release(s) due to file size restrictions in github.

## Preparing the Docker image.

* Pre-requisites
	* Git client
	* Docker

* Clone the repo:
	* `git clone https://github.com/WorldHealthOrganization/who-multi-bert-serving-gpu.git`
	* `cd who-multi-bert-serving-gpu`

* Build the image:
	* `image=<your-image-name`
	* `docker build -t ${image}`

## Run the Image

* `nvidia-docker run -it -p 8000:8000 ${image}`


## APIs

***Ping***
----

  Returns a status 200 response to confirm if the container is healthy.

* **URL**

  /ping/

* **Method:**

  `GET`
  
*  **URL Params**

   **Required:**
 
   None

* **Data Params**

  None

* **Success Response:**

  * **Code:** 200 <br />
    **Content:** `{"status": "ok"}`
 
* **Error Response:**

  * **Code:** 500 ERROR <br />
    **Content:** `{"status": "error"}`

* **Sample Call:**

  ```curl http://localhost:8000/ping
  ```


***predict***
----

  Returns the message payload appended with class probabilities and confidence score. This method expects a single json object containing a 'emm_text_text' field.

* **URL**

  /predict/

* **Method:**

  `POST`
  
*  **URL Params**

  None

* **Data Params**

  None

* **Success Response:**

  * **Code:** 200 <br />
    **Content:** `{ emm_text_text : "This is a sample news article.", probability : [0.14614616334438324, 0.0013145353877916932, 0.0014566172612830997, 0.07813428342342377, 8.907107257982716e-05, 0.00015411236381623894, 0.002774674678221345, 0.7479593753814697, 0.00978120043873787, 0.005860702600330114, 0.0063293748535215855], confidence: 0.57360524}`
 
* **Error Response:**

  * **Code:** 415 UNSUPPORTED MEDIA TYPE <br />
    **Content:** `This predictor only supports Json data`

  OR

  * **Code:** 500 ERROR <br />
    **Content:** `{"status": "error"}`

* **Sample Call:**

  ```curl --data-binary @${payload} -H "Content-Type: ${content}" -v http://localhost:8000/predict
  ```

***invocations***
----

  Returns the message payload appended with class probabilities and confidence score for each news article in the list. This method expects a list of json objects each containing a 'emm_text_text' field.

* **URL**

  /invocations/

* **Method:**

  `POST`
  
*  **URL Params**

  None

* **Data Params**

  None

* **Success Response:**

  * **Code:** 200 <br />
    **Content:** `[{"emm_text_text": "This is a test article", "emm_text": "blah", "probability": "[0.14614616334438324, 0.0013145353877916932, 0.0014566172612830997, 0.07813428342342377, 8.907107257982716e-05, 0.00015411236381623894, 0.002774674678221345, 0.7479593753814697, 0.00978120043873787, 0.005860702600330114, 0.0063293748535215855]", "confidence": "0.8046068"}, {"emm_text_text": "This is the second test article", "emm_text": "foo", "probability": "[0.24730996787548065, 0.0007172985933721066, 0.005345039069652557, 0.051877908408641815, 0.00040469400119036436, 0.0016399786109104753, 0.004078717902302742, 0.5800024271011353, 0.07807986438274384, 0.016841603443026543, 0.013702521100640297]", "confidence": "0.57360524"}]`
 
* **Error Response:**

  * **Code:** 415 UNSUPPORTED MEDIA TYPE <br />
    **Content:** `This predictor only supports Json data`

  OR

  * **Code:** 500 ERROR <br />
    **Content:** `{"status": "error"}`

* **Sample Call:**

  ```curl --data-binary @${payload} -H "Content-Type: ${content}" -v http://localhost:8000/invocations
  ```

## Data Pre-Processing

The model only accepts content in English. Additionally, all special characters need to be removed before submitting a request to the predict and invocations API. The WHO pipeline involves getting news feeds in XML format, which are expected to be transformed into JSON with the 'emm_text_text' field containing the article content. 