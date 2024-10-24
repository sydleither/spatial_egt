package SpatialEGT;

import java.io.*;
import java.nio.file.Paths;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

public class SpatialEGT {
    public static void main(String[] args) {
        // read in arguments
        String dataDir = null;
        String expDir = null;
        String expName = null;
        String dimension = null;
        String rep = null;
        int visualizationFrequency = 0;
        if (args.length == 0) {
            dataDir = "output";
            expDir = "test";
            expName = "test";
            dimension = "2D";
            rep = "0";
        }
        else if (args.length == 5) {
            dataDir = args[0];
            expDir = args[1];
            expName = args[2];
            dimension = args[3];
            rep = args[4];
        }
        else if (args.length == 6) {
            dataDir = args[0];
            expDir = args[1];
            expName = args[2];
            dimension = args[3];
            rep = args[4];
            visualizationFrequency = Integer.parseInt(args[5]);
        }
        else {
            System.out.println("Please provide the following arguments: experiment directory, experiment name, dimension, replicate/seed, and (optional) visualization frequency.");
        }
        String saveLoc = dataDir+"/"+expDir+"/"+expName+"/"+rep+"/"+dimension;
        
        // read in json parameters
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> params;
        try{
            params = mapper.readValue(Paths.get(dataDir+"/"+expDir+"/"+expName+"/"+expName+".json").toFile(), Map.class);
        }
        catch (Exception e) {
            return;
        }

        // run models
        if (dimension.equals("2D")) {
            new SpatialEGT2D(saveLoc, params, Long.parseLong(rep), visualizationFrequency);
        }
        else if (dimension.equals("3D")) {
            new SpatialEGT3D(saveLoc, params, Long.parseLong(rep));
        }
        else if (dimension.equals("WM") || dimension.equals("0D")) {
            new SpatialEGT0D(saveLoc, params, Long.parseLong(rep));
        }
    }
}
