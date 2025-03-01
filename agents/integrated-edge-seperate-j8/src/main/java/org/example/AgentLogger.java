package org.example;

import java.util.Objects;

public class AgentLogger {
    public static void logCGEdge(String className, String methodName, String desc, String owner, String name, String descriptor) {

        String message = "AgentLogger|CG_edge: " + className.replace('/', '.') + "." + methodName + " " + desc +
                         " -> " + owner.replace('/', '.') + "." + name + " " + descriptor;
        System.out.println(message);
    }

    public static void logBranch(String className, String methodName, String desc, int ifCounter) {
        String ifStatementId = className.replace('/', '.') + "." + methodName + " " + desc + ":IF#" + ifCounter;
        System.out.println("AgentLogger|BRANCH: " + ifStatementId);
    }

    // public static void logVariable(String className, String methodName, String desc, String varName, int varValue) {
    //     String message = "AgentLogger|VARIABLE: " + className.replace('/', '.') + "." + methodName + " " + desc + varName + " = " + varValue;
    //     System.out.println(message);
    // }

    // public static void logVariable(String className, String methodName, String desc, String varName, Object varValue) {
    //     String message = "AgentLogger|VARIABLE: " + className.replace('/', '.') + "." + methodName + " " + desc + varName + " = " +  Objects.toString(varValue, "null");
    //     System.out.println(message);
    // }


// A thread-local flag to prevent reentrant logging.
    private static final ThreadLocal<Boolean> isLogging = ThreadLocal.withInitial(() -> false);

    public static void logVariable(String className, String methodName, String desc, String varName, int varValue) {
        if (isLogging.get()) {
            System.out.println("AgentLogger|VARIABLE (recursive): " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + varValue);
            return;
        }
        isLogging.set(true);
        try {
            System.out.println("AgentLogger|VARIABLE: " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + varValue);
        } finally {
            isLogging.set(false);
        }
    }

    public static void logVariable(String className, String methodName, String desc, String varName, float varValue) {
        if (isLogging.get()) {
            System.out.println("AgentLogger|VARIABLE (recursive): " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + varValue);
            return;
        }
        isLogging.set(true);
        try {
            System.out.println("AgentLogger|VARIABLE: " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + varValue);
        } finally {
            isLogging.set(false);
        }
    }

    public static void logVariable(String className, String methodName, String desc, String varName, double varValue) {
        if (isLogging.get()) {
            System.out.println("AgentLogger|VARIABLE (recursive): " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + varValue);
            return;
        }
        isLogging.set(true);
        try {
            System.out.println("AgentLogger|VARIABLE: " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + varValue);
        } finally {
            isLogging.set(false);
        }
    }

    public static void logVariable(String className, String methodName, String desc, String varName, long varValue) {
        if (isLogging.get()) {
            System.out.println("AgentLogger|VARIABLE (recursive): " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + varValue);
            return;
        }
        isLogging.set(true);
        try {
            System.out.println("AgentLogger|VARIABLE: " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + varValue);
        } finally {
            isLogging.set(false);
        }
    }

    public static void logVariable(String className, String methodName, String desc, String varName, Object varValue) {
        if (isLogging.get()) {
            System.out.println("AgentLogger|VARIABLE (recursive): " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + safeRepresentation(varValue));
            return;
        }
        isLogging.set(true);
        try {
            String representation = safeRepresentation(varValue);
            System.out.println("AgentLogger|VARIABLE: " 
                    + className.replace('/', '.') + "." + methodName + " " + desc + varName 
                    + " = " + representation);
        } finally {
            isLogging.set(false);
        }
    }

    // A safe fallback to avoid problematic toString calls (e.g., on recursive CFG objects).
    private static String safeRepresentation(Object varValue) {
        if (varValue == null) {
            return "null";
        }
        // Detect if the object's type might trigger recursion (e.g., WALA CFG classes)
        if (varValue.getClass().getName().contains("com.ibm.wala.cfg")) {
            return varValue.getClass().getName() + "@" 
                    + Integer.toHexString(System.identityHashCode(varValue));
        }
        try {
            return varValue.toString();
        } catch (Throwable t) {
            return varValue.getClass().getName() + "@" 
                    + Integer.toHexString(System.identityHashCode(varValue));
        }
    }


    public static void logBoundaries(String name, Object instruction, Object src, Object target){

         if (name.equals("processResolvedCall")) {  
            System.out.println("AgentLogger|addEdge: " + src + " " + instruction + " " + target);
        } else if (name.equals("visitInvokeInternal")) {
            System.out.println("AgentLogger|visitinvoke: " + instruction);
        }
    }
}