package com.example.fetchgui_learning_testing;

import androidx.appcompat.app.AppCompatActivity;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.ToggleButton;


import java.io.BufferedReader;
import java.io.ByteArrayInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.ByteBuffer;
import java.time.LocalTime;

public class MainActivity extends AppCompatActivity {

    EditText teachObjectText;
    EditText findObjectText;
    TextView textView;
//    TextView modeView;
    ImageView robotDisplay;
    ToggleButton teachingModeToggle;
    ToggleButton findingModeToggle;
    Thread responseThread;
    Thread displayThread;
    Thread clearInfoBoxThread;
    appMode mode;
    appMode requestedMode;
    Boolean hasTextViewTextChanged;
//    Boolean requestFindObject;
    int textViewLength;


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        teachObjectText = (EditText)findViewById(R.id.teach_object_text);
        findObjectText = (EditText)findViewById(R.id.find_object_text);
        textView = (TextView)findViewById(R.id.textView);
//        modeView = (TextView)findViewById(R.id.mode_view);
        robotDisplay = (ImageView) findViewById(R.id.robot_display);
        teachingModeToggle = (ToggleButton)findViewById(R.id.toggle_teaching_session);
        findingModeToggle = (ToggleButton)findViewById(R.id.toggle_finding_object);
        teachingModeToggle.setBackgroundColor(Color.LTGRAY);
        findingModeToggle.setBackgroundColor(Color.LTGRAY);
        mode = appMode.DEFAULT_MODE;
        requestedMode = null;
        hasTextViewTextChanged = false;
//        requestFindObject = false;
        textViewLength = 0;

        responseThread = new Thread(new ReceivingMessagesThread());
        displayThread = new Thread(new DisplayThread());
        clearInfoBoxThread = new Thread(new ClearInfo());
        responseThread.start();
        displayThread.start();
        clearInfoBoxThread.start();

    }
    class ReceivingMessagesThread implements Runnable{
        Socket s;
        ServerSocket ss;
        InputStreamReader isr;
        BufferedReader bufferedReader;

        String message;
        @Override
        public void run(){
            try{
                ss = new ServerSocket(5001);
                s = ss.accept();
                while(true)
                {
                    isr = new InputStreamReader(s.getInputStream());
                    bufferedReader = new BufferedReader(isr);
                    message = bufferedReader.readLine();
                    setViewText(message);
                }
            }catch(IOException e){
                System.out.println("Failed to accept");
                e.printStackTrace();
            }
        }
    }
    class DisplayThread implements Runnable{
        Socket s;
        ServerSocket ss;
        @Override
        public void run(){
            try{
                ss = new ServerSocket(5002);
                s = ss.accept();
                OutputStream sout = s.getOutputStream();
                DataInputStream sin = new DataInputStream(s.getInputStream());
//                BufferedInputStream sinb = new BufferedInputStream(sin)
                while(true)
                {
                    // Get length
                    byte[] size_buff = new byte[4];
                    sin.read(size_buff);
                    int size = ByteBuffer.wrap(size_buff).asIntBuffer().get();
                    // Send it back (?)
                    sout.write(size_buff);
                    // Create Image Buffer
                    byte[] img_buff = new byte[size];
//                  Receive Image bytes
                    sin.readFully(img_buff,0,size);
//                  Respond to mark completion of data read
                    sout.write(1);
//                  Convert bytes to a bitmap
                    ByteArrayInputStream bis = new ByteArrayInputStream(img_buff);
                    Bitmap bp = BitmapFactory.decodeStream(bis);
                    try {
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                robotDisplay.setImageBitmap(bp);
                            }
                        });
                    }catch(Exception e){
                        e.printStackTrace();
                    }
                }
            }catch(IOException e){
                System.out.println("Failed to accept");
                e.printStackTrace();
            }
        }
    }

    public void println(String print){
        System.out.println(print);
    }
    class ClearInfo implements Runnable {
        long prevTime = System.currentTimeMillis();
        int clearTime = 3000;

        @Override
        public void run() {
            int clearTime = 3000;
            while(true) {
//                If the text has not changed and the defined time has passed since the text has changed
//                Reset the text view.
                if (!hasTextViewTextChanged && (System.currentTimeMillis() - prevTime) > clearTime) {
                    textView.setText("Info: ");
                    prevTime = System.currentTimeMillis();
                } else if(hasTextViewTextChanged){ // If it has changed reset the timer and set that it has changed to false
                    prevTime = System.currentTimeMillis();
                    clearTime = (int) (1000*textViewLength/12.16);
                    if (clearTime < 3000) {
                        clearTime = 3000;
                    }
                    hasTextViewTextChanged = false;
                }
            }
        }
    }

    private appMode toggleToMode(RequestEnums toggle){
        switch (toggle) {
            case TOGGLE_TEACHING_MODE:{
                return appMode.TEACHING_MODE;
            }
            case TOGGLE_FINDING_MODE:{
                return appMode.FINDING_MODE;
            }
        }
        return appMode.DEFAULT_MODE;
    }
    public void cancel_button(View v) {
//        if (requestFindObject){
//            requestFindObject = false;
//            setViewText("Cancelled finding object");
//            vocalizeText("Cancelled finding object");
//        }
//        if(requestedMode != null){
//            switch(requestedMode){
//                case FINDING_MODE:{
//                    requestedMode = null;
//                    setViewText("Cancelled Finding Mode");
//                    setMode(appMode.DEFAULT_MODE);
//                }
//                default: {
//                    requestedMode = null;
//                }
//            }
//        }
    }
    public void ok_button(View v) {
//        if (requestFindObject){
//            setViewText("The robot will attempt to find " + findObjectText.getText().toString() + ".");
//            send(Integer.toString(RequestEnums.FIND_OBJECT.getValue()), findObjectText.getText().toString().replace(" ", "").toLowerCase());
//            requestFindObject = false;
//        }
//        if(requestedMode != null){
//            switch(requestedMode){
//                case FINDING_MODE:{
//                    requestedMode = null;
//                    setMode(appMode.FINDING_MODE);
//                    setViewText("Entering Finding Mode");
//                }
//                default: {
//                    requestedMode = null;
//                }
//            }
//        }
    }
    public void requestToggleMode(RequestEnums modeSetting) {
        appMode temporaryMode = toggleToMode(modeSetting);
//        Reset if the current mode is the toggle button pressed is the same as the current mode
        if (mode == temporaryMode){
            setMode(appMode.DEFAULT_MODE);
        } else {
//            switch (temporaryMode) {
//                case FINDING_MODE: {
//                    if(requestedMode == null) {
//                        setMode(appMode.DEFAULT_MODE);
//                        findingModeToggle.setChecked(true);
//                        setViewText("Are you sure you want to enter Finding Mode.  Click OK to continue or Cancel to not enter this mode");
//                        vocalizeText("Are you sure you want to enter Finding Mode.  Click OK to continue or Cancel to not enter this mode");
//                        requestedMode = appMode.FINDING_MODE;
//                        break;
//                    }
//                    setMode(appMode.DEFAULT_MODE);
//                    requestedMode = null;
//                    break;
//                }
//                default: {
                    setMode(temporaryMode);
//                }
            }
//        }
    }
    public void resetToggleRequests(){
        requestedMode = null;
    }
    private void setMode(appMode sentMode) {
        send(Integer.toString(sentMode.getValue()),"");

        appMode prevMode = mode;
        if (mode == sentMode){
            mode = appMode.DEFAULT_MODE;
        } else {
            mode = sentMode;
        }
//        Send mode in here and wait for update message adjust mode if need be
        switch (mode) {
            case TEACHING_MODE: {
                teachingModeToggle.setChecked(true);
                findingModeToggle.setChecked(false);
                teachingModeToggle.setBackgroundColor(Color.GREEN);
                findingModeToggle.setBackgroundColor(Color.LTGRAY);
                break;
            }
            case FINDING_MODE: {
                teachingModeToggle.setChecked(false);
                findingModeToggle.setChecked(true);
                teachingModeToggle.setBackgroundColor(Color.LTGRAY);
                findingModeToggle.setBackgroundColor(Color.GREEN);
                break;
            }
            case DEFAULT_MODE: {
                teachingModeToggle.setChecked(false);
                findingModeToggle.setChecked(false);
                teachingModeToggle.setBackgroundColor(Color.LTGRAY);
                findingModeToggle.setBackgroundColor(Color.LTGRAY);
                break;
            }
        }
//        switch(mode){
//            case FINDING_MODE: {
//                modeView.setText("Finding");
//                break;
//            }
//            case TEACHING_MODE: {
//                modeView.setText("Teaching");
//                break;
//            }
//            case DEFAULT_MODE: {
//                modeView.setText("Default");
//                break;
//            }
//        }

        if(mode == appMode.DEFAULT_MODE && prevMode != appMode.DEFAULT_MODE){
            setViewText("Left " + ((prevMode == appMode.TEACHING_MODE)?"Teaching Mode":"Finding Mode" + "."));
            vocalizeText("Left " + ((prevMode == appMode.TEACHING_MODE)?"Teaching Mode":"Finding Mode" + "."));
        }
        else if (mode == appMode.DEFAULT_MODE){
            setViewText("Reset Modes.");
        }
        else if(mode == sentMode){
            if (mode == appMode.TEACHING_MODE){
                setViewText("Entered Teaching Mode. You can start teaching me objects.");
                vocalizeText("Entered Teaching Mode. You can start teaching me objects.");
            }
            else if (mode == appMode.FINDING_MODE){
                setViewText("Entered Finding Mode.");
                vocalizeText("Entered Finding Mode.");
            }
        }
//        else if(?) {
//            setViewText("Failed to enter requested mode. Entering " + ((mode == appMode.TEACHING_MODE)?"Teaching Mode":(mode == appMode.FINDING_MODE)?"Finding Mode":"Default Mode"));
//        }
        resetToggleRequests();
    }
    private void send(String prefix, String body){
        MessageSender messageSender = new MessageSender();
        messageSender.execute(prefix+"|"+body);
    }
    private void setViewText(String message){
        textViewLength = message.length();
        textView.setText("Info: " + message);
        hasTextViewTextChanged = true;
    }
//    Sends the text to the robot to be spoke
    private void vocalizeText(String message) {
        send(Integer.toString(RequestEnums.RESPONSE.getValue()), message);
    }
    public void toggleTeachingMode(View v) {
        if(mode == appMode.DEFAULT_MODE || mode == appMode.TEACHING_MODE) {
            requestToggleMode(RequestEnums.TOGGLE_TEACHING_MODE);
        }
        else{
            setViewText("Must leave current mode before entering Teaching Mode");
            vocalizeText("Must leave current mode before entering Teaching Mode");
            teachingModeToggle.setChecked(false);
        }
    }
    public void toggleFindingMode(View v) {
        if(mode == appMode.DEFAULT_MODE || mode == appMode.FINDING_MODE){
            requestToggleMode(RequestEnums.TOGGLE_FINDING_MODE);
        }
        else{
            setViewText("Must leave current mode before entering Finding Mode");
            vocalizeText("Must leave current mode before entering Finding Mode");
//            Might be useless now
            findingModeToggle.setChecked(false);
        }
    }
    public void saveTeachingObject(View v){
        if(mode == appMode.TEACHING_MODE) {
            if (!teachObjectText.getText().toString().isEmpty()) {
                send(Integer.toString(RequestEnums.SAVE_TEACHING_OBJECT.getValue()), teachObjectText.getText().toString().replace(" ", "").toLowerCase());
                setViewText(teachObjectText.getText().toString() + " has been saved.");
                vocalizeText(teachObjectText.getText().toString() + " has been saved.");
            }
            else {
                setViewText("Enter a valid object name");
                vocalizeText("Enter a valid object name");
            }
        }else{
            setViewText("Cannot save object, please enter Teaching Mode before saving.");
            vocalizeText("Cannot save object, please enter Teaching Mode before saving.");
        }
    }
    public void findObject(View v){
        if(mode == appMode.FINDING_MODE) {
//            requestFindObject = true;
//            setViewText("Are you at a safe distance from the robot? Click OK to continue.");
//            vocalizeText("Are you at a safe distance from me? Click OK to continue.");
            if (!findObjectText.getText().toString().isEmpty()) {
                setViewText("The robot will attempt to find " + findObjectText.getText().toString() + ".");
                send(Integer.toString(RequestEnums.FIND_OBJECT.getValue()), findObjectText.getText().toString().replace(" ", "").toLowerCase());
            }
            else {
                setViewText("Enter a valid object name");
                vocalizeText("Enter a valid object name");
            }
//            vocalizeText(findObjectText.getText().toString() + " is being found.");
        }else{
            setViewText("Not in Finding Mode.  Please enter Finding Mode before trying to find objects.");
            vocalizeText("Not in Finding Mode.  Please enter Finding Mode before trying to find objects.");
        }
    }
}