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

        foreach (PixelInfo pixelInfo in pixelLocations) {
            GameObject pixel = Instantiate(this.Pixel);
            pixel.transform.SetParent(this.transform);
            pixel.transform.localPosition = new Vector3(pixelInfo.x, pixelInfo.y, pixelInfo.z);
            pixel.transform.localScale = new Vector3(this.pixelSize, this.pixelSize, this.pixelSize);
            this.pixels.Add(pixel);
        }
    }

    private PixelInfo[] getMask()
    {
        HttpWebRequest request = (HttpWebRequest)WebRequest.Create("http://127.0.0.1:5000/mask");
        HttpWebResponse response = (HttpWebResponse)request.GetResponse();
        StreamReader reader = new StreamReader(response.GetResponseStream());
        string jsonResponse = reader.ReadToEnd();
        Debug.Log(jsonResponse);
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

    public PixelInfo (float x, float y, float z)
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }
}