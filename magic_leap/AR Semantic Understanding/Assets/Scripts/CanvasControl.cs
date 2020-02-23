using System;
using System.Collections;
using System.Collections.Generic;
using System.Net;
using System.IO;
using UnityEngine;

public class CanvasControl : MonoBehaviour
{
    public GameObject Pixel;

    private List<GameObject> pixels;

    /*
    private PixelInfo[] pixelLocations = {
        new PixelInfo(0.3f, 0.3f, 2f),
        new PixelInfo(0.0f, 0.0f, 2f),
        new PixelInfo(-0.3f, -0.3f, 2f)
    };
    */

    // Pixel width/height/depth
    private float pixelSize = 0.1f;

    // Start is called before the first frame update
    void Start()
    {
        this.pixels = new List<GameObject>();
    }

    // Update is called once per frame
    void Update()
    {
        foreach (GameObject pixel in this.pixels) {
            Destroy(pixel);
        }

        this.pixels.Clear();

        PixelInfo[] pixelLocations = this.getMask();

        Debug.Log("Updates");

        Debug.Log(string.Format("Num labels: {0}", pixelLocations.Length));

        int count = 0;
        foreach (PixelInfo pixelInfo in pixelLocations) {
            if (count > 3) {
                Debug.Log("break");
                //break;
            }
            GameObject pixel = Instantiate(this.Pixel);
            pixel.transform.SetParent(this.transform);
            float x = pixelInfo.x / 224.0f;
            float y = pixelInfo.y / 224.0f;
            float z = pixelInfo.z + 2;
            Debug.Log(string.Format("{0}, {1}, {2}", x, y, z));
            pixel.transform.localPosition = new Vector3(x, y, z);
            pixel.transform.localScale = new Vector3(this.pixelSize, this.pixelSize, this.pixelSize);
            this.pixels.Add(pixel);
            count++;
        }
    }

    private PixelInfo[] getMask()
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://10.197.154.206:5005/mask");
        HttpWebResponse response = (HttpWebResponse)request.GetResponse();
        StreamReader reader = new StreamReader(response.GetResponseStream());
        string jsonResponse = reader.ReadToEnd();
        //Debug.Log(jsonResponse);
        LocationData locationData = JsonUtility.FromJson<LocationData>(jsonResponse);
          
        return locationData.locations;
    }
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
    public string color;

    public PixelInfo(float x, float y, float z, string color)
    {
        this.x = x;
        this.y = y;
        this.z = z;
        this.color = color;
    }
}