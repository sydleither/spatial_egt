package SpatialEGT;

import java.nio.file.Paths;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

public class SpatialEGT {
    public static void main(String[] args) {
        // read in arguments
        String expDir = args[0];
        String expName = args[1];
        String dimension = args[2];
        String rep = args[3];
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
