import tensorflow as tf

model=tf.keras.models.load_model('final_model.h5')

def predict(left_eye,right_eye):
    score1=model.predict((left_eye)/255,verbose=0)
    score2=model.predict((right_eye)/255,verbose=0)
    if score1<0.5 and score2<0.5:
        return False
    return True