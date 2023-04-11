package com.example.fetchgui_learning_testing;

public enum RequestEnums {
    TOGGLE_TEACHING_MODE(0),
    TOGGLE_FINDING_MODE(1),
    RESET_MODE(2),
    SAVE_TEACHING_OBJECT(3),
    FIND_OBJECT(4),
    OK(5),
    RESPONSE(6);

    private int value;

    RequestEnums(int i) {
        this.value = i;
    }

    public void setValue(int i) {this.value = i;}
    public int getValue() {
        return value;
    }
}
