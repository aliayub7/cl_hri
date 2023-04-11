import rospy
from sound_play.msg import SoundRequest
from sound_play.libsoundplay import SoundClient


class SpeechControl(object):
    """ Robot Speech interface """

    def __init__(self): #rosrun sound_play soundplay_node.py needs to run on the robot!
        self.soundhandle = SoundClient()
        rospy.sleep(1)

    def say(self, sentence, voice = 'voice_rab_diphone', volume=1.0):
        #voice = 'voice_ked_diphone', 'voice_rab_diphone', 'voice_kal_diphone', 'voice_don_diphone'
        self.soundhandle.say(sentence, voice, volume)

if __name__ == '__main__':
    import time
    rospy.init_node("test_speech", anonymous=True)
    speech_module = SpeechControl()
    speak_string = "Hello, I am Fetch, your butler for the day. How may I help you?"
    print (len(speak_string))
    speech_module.say(speak_string)
    speech_module.say("                                    ...")
    #rospy.sleep(len(speak_string)/11.9)
    #speech_module.soundhandle.stopAll()
