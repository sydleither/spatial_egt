package SpatialEGT;

import java.io.*;
import java.nio.file.Paths;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

public class SpatialEGT {
    public static void main(String[] args) {
        // read in arguments
        String expDir = null;
        String expName = null;
        String dimension = null;
        String rep = null;
        if (args.length == 0) {
            expDir = "test";
            expName = "test";
            dimension = "2D";
            rep = "0";
        }
        else if (args.length == 4) {
            expDir = args[0];
            expName = args[1];
            dimension = args[2];
            rep = args[3];
        }
        else {
            System.out.println("Please provide the following arguments: experiment directory, experiment name, dimension, and replicate/seed.");
        }
        String saveLoc = "output/"+expDir+"/"+expName+"/"+rep+"/"+dimension;
        
        // read in json parameters
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> params;
        try{
            params = mapper.readValue(Paths.get("output/"+expDir+"/"+expName+"/"+expName+".json").toFile(), Map.class);
        }
        catch (Exception e) {
            return;
        }

        // run models
        if (dimension.equals("2D")) {
            new SpatialEGT2D(saveLoc, params, Long.parseLong(rep));
        }
        else if (dimension.equals("3D")) {
            new SpatialEGT3D(saveLoc, params, Long.parseLong(rep));
        }
        else if (dimension.equals("WM") || dimension.equals("0D")) {
            new SpatialEGT0D(saveLoc, params, Long.parseLong(rep));
        }
    }
}
