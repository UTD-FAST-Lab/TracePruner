package org.example;

import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class InvocationLoggerBatch {

    // === Trace Management ===
    private static final int BATCH_SIZE = 20_000_000;
    private static List<String> currentBatch = new ArrayList<>(BATCH_SIZE);
    private static final Map<Integer, File> batchFiles = new ConcurrentHashMap<>();
    private static long globalLineCounter = -1;
    private static int batchId = 0;

    private static final BitSet excludedIndices = new BitSet();
    private static final Map<String, Long> instruction2start = new HashMap<>();

    private static int edgeCounter = 0;
    private static BufferedWriter edgesWriter;

    private static final Pattern OFFSET_PATTERN = Pattern.compile("@(\\d+)");
    private static final Pattern NODE_PATTERN = Pattern.compile("Node: < (?:Primordial|Application), ([^,]+), ([^ ]+) >");

    private static String programPath;

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

    public static void close() {
        try {
            if (!currentBatch.isEmpty()) {
                flushCurrentBatch();
            }
            if (edgesWriter != null) edgesWriter.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void flushCurrentBatch() throws IOException {
        File file = new File(programPath + "/trace_batch_" + batchId + ".txt");
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(file))) {
            for (String line : currentBatch) {
                writer.write(line);
                writer.newLine();
            }
        }
        batchFiles.put(batchId, file);
        currentBatch.clear();
        batchId++;
    }

    private static String reformatNodeString(String nodeString) {
        Matcher matcher = NODE_PATTERN.matcher(nodeString);
        if (matcher.find()) {
            String className = matcher.group(1);
            String methodSignature = matcher.group(2);
            if (className.startsWith("L")) className = className.substring(1);
            int parenIndex = methodSignature.indexOf('(');
            if (parenIndex != -1) {
                String methodName = methodSignature.substring(0, parenIndex);
                String params = methodSignature.substring(parenIndex + 1);
                return className + "." + methodName + ":(" + params;
            }
        }
        return null;
    }

    public static void addInstruction(Object instruction) {
        if (globalLineCounter == -1) {
            globalLineCounter = 0;
        }
        instruction2start.put(instruction.toString(), globalLineCounter);
    }

    public static void addLineToTrace(String className, String methodName, String desc, String owner, String name, String descriptor) {
        if (globalLineCounter == -1) {
            return;
        }
        currentBatch.add(className + "." + methodName + " " + desc + "," + owner + "." + name + " " + descriptor);
        globalLineCounter++;
        if (currentBatch.size() >= BATCH_SIZE) {
            try {
                flushCurrentBatch();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void writeTrace(Object instruction, Object src, Object target) {
        String instructionStr = instruction.toString();
        Matcher matcher = OFFSET_PATTERN.matcher(instructionStr);
        int offset = matcher.find() ? Integer.parseInt(matcher.group(1)) : -1;
        if (offset == -1 || !instruction2start.containsKey(instructionStr)) return;

        long start = instruction2start.get(instructionStr);
        long end = globalLineCounter;

        if (instructionStr.contains(", Ljava/lang")) {
            for (long i = start; i <= end; i++) excludedIndices.set((int) i);
            return;
        }

        List<String> filteredTrace = new ArrayList<>();
        for (long i = start; i < end; i++) {
            if (!excludedIndices.get((int) i)) {
                filteredTrace.add(readTraceLine(i));
            }
        }

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(programPath + "/" + edgeCounter + ".txt"))) {
            writer.write("Edge: " + src + "," + offset + "," + target);
            writer.newLine();
            for (String line : filteredTrace) {
                writer.write(line);
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

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

        try {
            edgesWriter.write(edgeCounter + "," + srcStr + "," + offset + "," + targetStr);
            edgesWriter.newLine();
            edgesWriter.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
        edgeCounter++;
    }

    private static String readTraceLine(long lineIndex) {
        int batchIndex = (int) (lineIndex / BATCH_SIZE);
        int lineInBatch = (int) (lineIndex % BATCH_SIZE);
    
        if (batchIndex == batchId) {
            // Still in memory
            return currentBatch.get(lineInBatch);
        }
    
        File batchFile = batchFiles.get(batchIndex);
        if (batchFile == null) return null;
    
        try (BufferedReader reader = new BufferedReader(new FileReader(batchFile))) {
            for (int i = 0; i < lineInBatch; i++) reader.readLine();
            return reader.readLine();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
