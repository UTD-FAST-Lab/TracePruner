package org.example;

import java.io.*;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class InvocationLoggerMemory {

    // === Data Structures ===
    private static ArrayList<String> trace = new ArrayList<>();  // full trace kept in memory
    private static BitSet excludedIndices = new BitSet();        // tracks excluded trace lines
    private static HashMap<String, Long> instruction2start = new HashMap<>();

    private static long lineCounter = -1;
    private static long MAX_TRACE_SIZE = 100_000_000;
    private static int edgeCounter = 0;
    private static BufferedWriter edgesWriter;

    private static final Pattern OFFSET_PATTERN = Pattern.compile("@(\\d+)");
    private static final Pattern NODE_PATTERN = Pattern.compile("Node: < (?:Primordial|Application), ([^,]+), ([^ ]+) >");

    private static String programPath;

    // === Initialize Writers ===
    public static void initialize(String programDirectory) {
        programPath = programDirectory;
        try {
            edgesWriter = new BufferedWriter(new FileWriter(programPath + "/edges.csv", true));
            File file = new File(programPath + "/edges.csv");
            if (file.length() == 0) {
                edgesWriter.write("edge_id,method,offset,target");
                edgesWriter.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // === Close Writers ===
    public static void close() {
        try {
            if (edgesWriter != null) {
                edgesWriter.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    // === Reformat Node String ===
    private static String reformatNodeString(String nodeString) {
        Matcher matcher = NODE_PATTERN.matcher(nodeString);

        if (matcher.find()) {
            String className = matcher.group(1);
            String methodSignature = matcher.group(2);

            if (className.startsWith("L")) {
                className = className.substring(1);
            }

            int parenIndex = methodSignature.indexOf('(');
            if (parenIndex != -1) {
                String methodName = methodSignature.substring(0, parenIndex);
                String params = methodSignature.substring(parenIndex + 1);
                return className + "." + methodName + ":(" + params;
            }
        }
        return null;
    }

    // === Add Instruction Start ===
    public static void addInstruction(Object instruction) {
        String instructionStr = instruction.toString();

        if (lineCounter == -1) 
            lineCounter = 0; 

        instruction2start.put(instructionStr, lineCounter);
    }

    // === Add Line to In-Memory Trace ===
    public static void addLineToTrace(String className, String methodName, String desc, String owner, String name, String descriptor) {

        if (lineCounter == -1 || lineCounter > MAX_TRACE_SIZE) {
            return; // No trace to add to
        }

        else if (lineCounter == MAX_TRACE_SIZE) {
            System.out.println("Trace size limit reached. Stopping trace collection.");
            return;      
        }

        else {
            trace.add(className + "." + methodName + " " + desc + "," + owner + "." + name + " " + descriptor);
            lineCounter++;
        }
    }

    
    public static void addLineToTrace(String className, String methodName, String desc,  int ifCounter, boolean isIFBranch) {
        // add line to the trace, increament the line counter
        if (lineCounter == -1 || lineCounter > MAX_TRACE_SIZE) {
            return; // No trace to add to
        }
        else if (lineCounter == MAX_TRACE_SIZE) {
            System.out.println("Trace size limit reached. Stopping trace collection.");
            return;      
        }
        else {
            String ifStatementId = className + "." + methodName + " " + desc;
            if (isIFBranch)
                ifStatementId +=  ":IF#" + ifCounter;
            else
                ifStatementId +=  ":ELSE#" + ifCounter;

            trace.add(ifStatementId);
            lineCounter++;  
        }    
    }

    // === Write Edge Trace ===
    public static void writeTrace(Object instruction, Object src, Object target) {
        String instructionStr = instruction.toString();
        Matcher matcher = OFFSET_PATTERN.matcher(instructionStr);

        int offset = -1;
        if (matcher.find()) {
            offset = Integer.parseInt(matcher.group(1));
        } else {
            return; // No program counter found
        }

        if (instruction2start.containsKey(instructionStr)) {
            long start = instruction2start.get(instructionStr);
            long end = lineCounter;

            if (instructionStr.contains(", Ljava/lang")) {
                // Mark excluded
                for (long i = start; i <= end; i++) {
                    excludedIndices.set((int) i);
                }
            } else {
                // Filter trace lines
                List<String> filteredTrace = new ArrayList<>();
                for (long i = start; i < end; i++) {
                    if (!excludedIndices.get((int) i)) {
                        filteredTrace.add(trace.get((int) i));
                    }
                }

                // Write filtered trace to a file
                try (BufferedWriter edgeWriter = new BufferedWriter(new FileWriter(programPath + "/" + edgeCounter + ".txt"))) {
                    edgeWriter.write("Edge: " + src + "," + offset + "," + target);
                    edgeWriter.newLine();
                    for (String line : filteredTrace) {
                        edgeWriter.write(line);
                        edgeWriter.newLine();
                    }
                } catch (IOException e) {
                    e.printStackTrace();
                }

                // Reformat src and target
                String srcStr = src.toString();
                String targetStr = target.toString();
                if (srcStr.contains("fakeRootMethod") || srcStr.contains("fakeWorldClinit")) {
                    srcStr = "<boot>";
                    offset = 0;
                } else {
                    srcStr = reformatNodeString(srcStr);
                }
                targetStr = reformatNodeString(targetStr);

                if (srcStr == null || targetStr == null) return;

                // Write to edges.csv
                try {
                    edgesWriter.write(edgeCounter + "," + srcStr + "," + offset + "," + targetStr);
                    edgesWriter.newLine();
                    edgesWriter.flush();
                } catch (IOException e) {
                    e.printStackTrace();
                }

                edgeCounter++;
            }
        }
    }
}
