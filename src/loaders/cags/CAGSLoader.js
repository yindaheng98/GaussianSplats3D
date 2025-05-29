import * as THREE from 'three';
import * as CAGS from 'cags.js';
import { delayedExecute, nativePromiseWithExtractedComponents } from '../../Util.js';
import { SplatBuffer } from '../SplatBuffer.js';
import { SplatBufferGenerator } from '../SplatBufferGenerator.js';
import { LoaderStatus } from '../LoaderStatus.js';
import { CAGSParser } from './CAGSParser.js';

function finalize(splatData, optimizeSplatData, minimumAlpha, compressionLevel, sectionSize, sceneCenter, blockSize, bucketSize) {
    if (optimizeSplatData) {
        const splatBufferGenerator = SplatBufferGenerator.getStandardGenerator(minimumAlpha, compressionLevel,
                                                                               sectionSize, sceneCenter,
                                                                               blockSize, bucketSize);
        return splatBufferGenerator.generateFromUncompressedSplatArray(splatData);
    } else {
        return SplatBuffer.generateFromUncompressedSplatArrays([splatData], minimumAlpha, 0, new THREE.Vector3());
    }
}

export class CAGSLoader {

    static loadFromURL(fileName, onProgress, progressiveLoadToSplatBuffer, onProgressiveLoadSectionProgress,
                       minimumAlpha, compressionLevel, optimizeSplatData = true, outSphericalHarmonicsDegree = 0,
                       headers, sectionSize, sceneCenter, blockSize, bucketSize) {

        const loadPromise = nativePromiseWithExtractedComponents();

        // Extract base path from fileName for loading related files
        const basePath = fileName.substring(0, fileName.lastIndexOf('/'));

        if (onProgress) onProgress(0, '0%', LoaderStatus.Downloading);

        // Start the CAGS loading process
        CAGSLoader.loadCAGSData(basePath, onProgress, outSphericalHarmonicsDegree, headers)
            .then((splatArray) => {
                if (onProgress) onProgress(100, '100%', LoaderStatus.Done);

                return delayedExecute(() => {
                    return finalize(splatArray, optimizeSplatData, minimumAlpha, compressionLevel,
                                    sectionSize, sceneCenter, blockSize, bucketSize);
                });
            })
            .then((splatBuffer) => {
                loadPromise.resolve(splatBuffer);
            })
            .catch((error) => {
                console.error('Error loading CAGS file:', error);
                loadPromise.reject(error);
            });

        return loadPromise.promise;
    }

    static async loadCAGSData(basePath, onProgress, outSphericalHarmonicsDegree, headers) {
        try {
            // Calculate total files to download for progress tracking
            const ATTRIBUTES_LIST = CAGSParser.getAttributesList();
            let totalFiles = 2; // base codebook + base codes
            for (const attr in ATTRIBUTES_LIST) {
                if (ATTRIBUTES_LIST.hasOwnProperty(attr)) {
                    totalFiles += ATTRIBUTES_LIST[attr] * 2; // codebook + codes for each layer
                }
            }

            let filesLoaded = 0;
            const updateProgress = (message) => {
                const percent = (filesLoaded / totalFiles) * 90; // Reserve 10% for processing
                if (onProgress) {
                    onProgress(percent, `${percent.toFixed(1)}%`, LoaderStatus.Downloading, message);
                }
            };

            // Create DeQuantizer instance
            const deQuantizer = new CAGS.DeQuantizer();

            try {
                // Load base layer codebook
                updateProgress('Loading base codebook...');
                const baseCodebookData = await CAGSLoader.fetchFileAsArrayBuffer(`${basePath}/point_cloud.codebook.npz`, headers);
                await deQuantizer.loadBaseLayerCodebook(new Uint8Array(baseCodebookData));
                filesLoaded++;
                updateProgress('Base codebook loaded');

                // Load enhancement layer codebooks
                for (const attr in ATTRIBUTES_LIST) {
                    if (ATTRIBUTES_LIST.hasOwnProperty(attr)) {
                        for (let i = 1; i <= ATTRIBUTES_LIST[attr]; i++) {
                            updateProgress(`Loading enhancement codebook: ${attr} layer ${i}/${ATTRIBUTES_LIST[attr]}`);
                            const url = `${basePath}/point_cloud.layer.${attr}.${i}.codebook.npz`;
                            const data = await CAGSLoader.fetchFileAsArrayBuffer(url, headers);
                            await deQuantizer.loadEnhancementLayerCodebook(attr, new Uint8Array(data));
                            filesLoaded++;
                            updateProgress(`Enhancement codebook loaded: ${attr} layer ${i}`);
                        }
                    }
                }

                // Load base layer codes
                updateProgress('Loading base codes...');
                const baseCodesData = await CAGSLoader.fetchFileAsArrayBuffer(`${basePath}/point_cloud.drc`, headers);
                await deQuantizer.loadBaseLayerCodes(new Uint8Array(baseCodesData));
                filesLoaded++;
                updateProgress('Base codes loaded');

                // Load enhancement layer codes
                for (const attr in ATTRIBUTES_LIST) {
                    if (ATTRIBUTES_LIST.hasOwnProperty(attr)) {
                        for (let i = 1; i <= ATTRIBUTES_LIST[attr]; i++) {
                            updateProgress(`Loading enhancement codes: ${attr} layer ${i}/${ATTRIBUTES_LIST[attr]}`);
                            const url = `${basePath}/point_cloud.layer.${attr}.${i}.codes.npz`;
                            const data = await CAGSLoader.fetchFileAsArrayBuffer(url, headers);
                            await deQuantizer.loadEnhancementLayerCodes(attr, new Uint8Array(data));
                            filesLoaded++;
                            updateProgress(`Enhancement codes loaded: ${attr} layer ${i}`);
                        }
                    }
                }

                // Perform dequantization
                if (onProgress) onProgress(90, '90%', LoaderStatus.Processing, 'Dequantizing data...');
                const startTime = performance.now();
                const attributes = await deQuantizer.dequantize();
                const endTime = performance.now();
                const dequantizeTime = endTime - startTime;

                console.log(`CAGS dequantization completed in ${dequantizeTime.toFixed(2)}ms`);

                // Get position data from deQuantizer
                const positions = deQuantizer.getPosition();
                const numPoints = deQuantizer.getnPoints();

                console.log(`CAGS loaded ${numPoints} points`);

                // Add position data to attributes for the parser
                attributes.position = positions;

                // Convert to UncompressedSplatArray
                if (onProgress) onProgress(95, '95%', LoaderStatus.Processing, 'Converting to splat array...');
                const splatArray = await CAGSParser.convertDeQuantizedAttributesToSplatArray(attributes, outSphericalHarmonicsDegree);

                // Clean up
                deQuantizer.dispose();

                // Dispose of TensorFlow tensors if they exist
                if (typeof attributes.scale?.dispose === 'function') attributes.scale.dispose();
                if (typeof attributes.rotation?.dispose === 'function') attributes.rotation.dispose();
                if (typeof attributes.opacity?.dispose === 'function') attributes.opacity.dispose();
                if (typeof attributes.features_dc?.dispose === 'function') attributes.features_dc.dispose();
                if (typeof attributes.features_rest?.dispose === 'function') attributes.features_rest.dispose();

                return splatArray;

            } finally {
                // Ensure cleanup even if an error occurs
                if (deQuantizer && typeof deQuantizer.dispose === 'function') {
                    deQuantizer.dispose();
                }
            }

        } catch (error) {
            console.error('Error in CAGS data loading:', error);
            throw error;
        }
    }

    static async fetchFileAsArrayBuffer(url, headers) {
        try {
            const fetchOptions = {};
            if (headers) fetchOptions.headers = headers;

            const response = await fetch(url, fetchOptions);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status} for URL: ${url}`);
            }
            return await response.arrayBuffer();
        } catch (error) {
            console.error(`Failed to fetch ${url}:`, error);
            throw error;
        }
    }
}
