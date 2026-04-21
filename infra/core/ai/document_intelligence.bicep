targetScope = 'resourceGroup'

@description('Tags that will be applied to all resources')
param tags object = {}

@description('Document Intelligence resource name')
param resourceName string

@description('AI Services account name for the project parent')
param aiServicesAccountName string = ''

@description('AI project name for creating the connection')
param aiProjectName string = ''

@description('Id of the user or app to assign application roles')
param principalId string

@description('Principal type of user or app')
param principalType string

@description('Name for the AI Foundry Document Intelligence connection')
param connectionName string = 'document-intelligence-connection'

@description('Location for all resources')
param location string = resourceGroup().location

// Get reference to the AI Services account and project to access their managed identities
resource aiAccount 'Microsoft.CognitiveServices/accounts@2025-04-01-preview' existing = if (!empty(aiServicesAccountName) && !empty(aiProjectName)) {
  name: aiServicesAccountName

  resource aiProject 'projects' existing = {
    name: aiProjectName
  }
}

// Azure AI Document Intelligence resource
resource documentIntelligence 'Microsoft.CognitiveServices/accounts@2024-10-01' = {
  name: resourceName
  location: location
  tags: tags
  kind: 'FormRecognizer'
  sku: {
    name: 'S0'
  }
  identity: {
    type: 'SystemAssigned'
  }
  properties: {
    publicNetworkAccess: 'Enabled'
    customSubDomainName: resourceName
    disableLocalAuth: true
    networkAcls: {
      defaultAction: 'Allow'
      virtualNetworkRules: []
      ipRules: []
    }
  }
}

// Role assignment for AI project managed identity to access Document Intelligence
resource aiProjectDocIntelligenceRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = if (!empty(aiServicesAccountName) && !empty(aiProjectName)) {
  scope: documentIntelligence
  name: guid(subscription().id, resourceGroup().id, 'doc-intel-sp-role', aiServicesAccountName, aiProjectName)
  properties: {
    principalId: (aiAccount::aiProject)!.identity.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908') // Cognitive Services User
  }
}

// Role assignment for the deploying user/service principal
resource userDocIntelligenceRoleAssignment 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: documentIntelligence
  name: guid(subscription().id, resourceGroup().id, principalId, 'doc-intel-user-role')
  properties: {
    principalId: principalId
    principalType: principalType
    roleDefinitionId: resourceId('Microsoft.Authorization/roleDefinitions', 'a97b65f3-24c7-4388-baec-2e87135dc908') // Cognitive Services User
  }
}

// Create the Document Intelligence connection to the AI Foundry project
module documentIntelligenceConnection './connection.bicep' = if (!empty(aiServicesAccountName) && !empty(aiProjectName)) {
  name: 'doc-intelligence-connection-creation'
  params: {
    aiServicesAccountName: aiServicesAccountName
    aiProjectName: aiProjectName
    connectionConfig: {
      name: connectionName
      category: 'CognitiveService'
      target: documentIntelligence.properties.endpoint
      authType: 'AAD'
      isSharedToAll: true
      metadata: {
        ApiType: 'Azure'
        ResourceId: documentIntelligence.id
        type: 'document_intelligence'
        Kind: 'FormRecognizer'
      }
    }
    credentials: {}
  }
  dependsOn: [
    aiProjectDocIntelligenceRoleAssignment
  ]
}

output documentIntelligenceName string = documentIntelligence.name
output documentIntelligenceEndpoint string = documentIntelligence.properties.endpoint
output documentIntelligenceConnectionName string = documentIntelligenceConnection!.outputs.connectionName
output documentIntelligenceConnectionId string = documentIntelligenceConnection!.outputs.connectionId
output documentIntelligenceResourceId string = documentIntelligence.id
