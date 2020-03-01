using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR.MagicLeap;

public class MeshGenerator : MonoBehaviour
{
    private Mesh mesh;
    
    private PixelInfo[] pixelLocations = new PixelInfo[0];
    private bool maskRequestMade = false;

    // Color of class labels
    // Class 0 is omitted so index with (class label - 1)
    private Color [] classColors = new Color[]
    {
        new Color(0.0f, 1.0f, 1.0f, 1.0f),
        new Color(1.0f, 0.0f, 1.0f, 1.0f),
        new Color(1.0f, 1.0f, 0.0f, 1.0f),
        new Color(1.0f, 0.0f, 0.0f, 1.0f),
        new Color(0.0f, 1.0f, 0.0f, 1.0f),
        new Color(0.0f, 0.0f, 1.0f, 1.0f),
        new Color(0.8f, 0.4f, 0.0f, 1.0f)
    };

    // Start is called before the first frame update
    void Start()
    {
        this.mesh = new Mesh();
        this.GetComponent<MeshFilter>().mesh = this.mesh;
    }

    // Update is called once per frame
    void Update()
    {
        UpdateMesh();
    }

    /*
    void CreateMesh()
    {
        Vector3[] vertices = new Vector3[]
        {
            //this.Camera.ViewportToWorldPoint(new Vector3(0.5f, 0.5f, 0.5f))
            //new Vector3(0.0f, -0.15f, 0.5f)
            //new Vector3(0, 0.05f, 0),
            //new Vector3(0.05f, 0, 0)
        };

        this.triangles = new int[]
        {
            0//, 1, 2
        };
    }
    */
    void UpdateMesh ()
    {
        // 0.370 is min ML display distance in z
        // canvas z offset to 0.4
        // x from -0.2 to 0.2 at z of 0.5
        // y from -0.15 to 0.15 at z of 0.5
        this.mesh.Clear();

        float z = 0.0f;
        float x_width = 0.065f; //0.6
        float y_width = 0.04125f; //0.04

        Vector3[] vertices = new Vector3[]
        {
            new Vector3(0.0f, 0.0f, z),  // mid point
            new Vector3(-x_width/2.0f, 0.0f, z),  // left point
            new Vector3(x_width/2.0f, 0.0f, z),  // right point
            new Vector3(0.0f, y_width/2.0f, z),  // top point
            new Vector3(0.0f, -y_width/2.0f, z), // bottom point
        };

        Color[] colors = new Color[]
        {
            new Color(1.0f, 0.0f, 0.0f, 1.0f),
            new Color(0.0f, 1.0f, 0.0f, 1.0f),
            new Color(0.0f, 0.0f, 1.0f, 1.0f),
            new Color(1.0f, 0.0f, 1.0f, 1.0f),
            new Color(0.0f, 1.0f, 1.0f, 1.0f)
        };

        this.mesh.vertices = vertices;
        this.mesh.colors = colors;
        //this.mesh.triangles = triangles;

        int[] indices = new int[]
        {
            0, 1, 2, 3, 4
        };

        this.mesh.SetIndices(indices, MeshTopology.Points, 0);
        

        return;
        /*
        this.mesh.Clear();

        //PixelInfo[] pixelLocations = this.getMask();

        //Debug.Log(string.Format("Num points: {0}", pixelLocations.Length));

        this.getMask();

        int numPoints = this.pixelLocations.Length;

        Vector3[] points = new Vector3[numPoints];
        int[] indices = new int[numPoints];
        Color[] colors = new Color[numPoints];
        */
        /*
        for(int i=0; i<points.Length; i++) {
            points[i] = new Vector3(
                UnityEngine.Random.Range(-0.2f, 0.2f),
                UnityEngine.Random.Range(-0.15f, 0.15f),
                UnityEngine.Random.Range (0.0f, 5.0f));
            indices[i] = i;
            colors[i] = new Color(
                UnityEngine.Random.Range(0.0f, 1.0f),
                UnityEngine.Random.Range(0.0f, 1.0f),
                UnityEngine.Random.Range(0.0f, 1.0f),
                1.0f);
        }
        */

        // 0.370 is min ML display distance in z
        // canvas z offset to 0.4
        // x from -0.2 to 0.2 at z of 0.5
        // y from -0.15 to 0.15 at z of 0.5

        /*
        int index = 0;
        foreach (PixelInfo pixelInfo in this.pixelLocations)
        {
            indices[index] = index;

            float x = ((pixelInfo.x / 560.0f) - 0.2f);
            float y = ((pixelInfo.y / 747.0f) - 0.15f);
            float z = (pixelInfo.z - 0.37f) / 2.0f - 0.1f;

            points[index] = new Vector3(x, -y, z);
            colors[index] = this.classColors[pixelInfo.pixelClass - 1];

            index++;
        }

        mesh.vertices = points;
        mesh.colors = colors;
        mesh.SetIndices(indices, MeshTopology.Points, 0);
        */
    }
    
    private async void getMask()
    {
        if (this.maskRequestMade) return;

        try
        {
            this.maskRequestMade = true;
            Task.Run(() => { makeMaskRequest(); });
        }
        catch (Exception)
        {
            this.maskRequestMade = false;
        }
    }

    public async Task makeMaskRequest()
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://192.168.1.18:5005/mask");
        HttpWebResponse response = (HttpWebResponse)request.GetResponse();
        StreamReader reader = new StreamReader(response.GetResponseStream());
        string jsonResponse = reader.ReadToEnd();
        LocationData locationData = JsonUtility.FromJson<LocationData>(jsonResponse);
        
        this.pixelLocations = locationData.locations;

        this.maskRequestMade = false;
    }

    /*
    private PixelInfo[] getMask()
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://192.168.1.18:5005/mask");
        HttpWebResponse response = (HttpWebResponse)request.GetResponse();
        StreamReader reader = new StreamReader(response.GetResponseStream());
        string jsonResponse = reader.ReadToEnd();
        LocationData locationData = JsonUtility.FromJson<LocationData>(jsonResponse);
          
        return locationData.locations;
    }
    */
}

[Serializable]
public class LocationData
{
    public PixelInfo[] locations;
}

[Serializable]
public class PixelInfo {
    public float x;
    public float y;
    public float z;
    public int pixelClass;

    public PixelInfo(float x, float y, float z, int pixelClass)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        this.pixelClass = pixelClass;
    }
}
