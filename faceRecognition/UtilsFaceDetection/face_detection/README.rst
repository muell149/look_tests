Onboarding Image API
--------

Set of modules used to collect a profile image and send it to a RabbitMQ queue
It detects if there is a face in the picture and if so, it send the image to a queue
to be processed later by the workers
