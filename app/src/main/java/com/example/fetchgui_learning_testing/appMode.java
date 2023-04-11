package com.example.fetchgui_learning_testing;

public enum appMode {
    TEACHING_MODE(0),
    FINDING_MODE(1),
    DEFAULT_MODE(2);

    private int value;

    appMode(int i) {
        this.value = i;
    }

    public void setValue(int i) {this.value = i;}
    public int getValue() {
        return value;
    }
}
